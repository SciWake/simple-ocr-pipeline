import torch
from torch import nn


class TokenDecoder(nn.Module):
    @torch.jit.export
    def get_input_size(self):
        return self.input_size

    @torch.jit.export
    def get_info(self):
        return self.info

    @torch.jit.export
    def get_vocab(self):
        return self.vocab

    @torch.jit.export
    def fp16(self):
        return self.fp16

    def __init__(
        self,
        input_size: int,
        tokens_count: int,
        rnn_type: str,
        num_layers: int,
        tail_hidden_linears: int,
        vocab_length: int,
        hidden_size: int = 1024,
        batch_first=False,
        fp16=False,
        info=None,
        bidirectional=False
    ):
        super(TokenDecoder, self).__init__()

        self.info = info
        self.input_size = input_size
        self.rnn_size = hidden_size
        self.fp16 = fp16
        self.rnn_num_directions = 2 if bidirectional else 1

        if rnn_type == 'GRU':
            rnn_type = nn.GRU
        elif rnn_type == 'LSTM':
            rnn_type = nn.LSTM
        else:
            raise NotImplementedError(f'Decoder {rnn_type} is not supported yet')

        self.rnn = rnn_type(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=batch_first,
            bidirectional=bidirectional,
        )

        # fcnn после rnn части
        self.tail_fcnn = []
        linear_in_size = hidden_size * self.rnn_num_directions
        for tail_hidden_size in tail_hidden_linears or []:
            self.tail_fcnn.extend([
                nn.Linear(linear_in_size, tail_hidden_size),
                nn.ReLU()
            ])
            linear_in_size = tail_hidden_size

        self.tail_fcnn.append(nn.Linear(linear_in_size, vocab_length + 1))  # +1 для разделителя
        self.output = nn.Sequential(*self.tail_fcnn)
        self.softmax = nn.LogSoftmax(dim=2)

        self.output = nn.Linear(hidden_size * self.rnn_num_directions, tokens_count)

        self.init_rcnn(self.rnn)
        self.init_rcnn(self.output)

        self.register_backward_hook(self.backward_hook)

    def forward(self, x):
        x, h = self.rnn(x)  # [w, b, h * c]

        x = self.output(x)
        x = self.softmax(x)

        return x

    def get_trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

    def init_rcnn(self, m):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.xavier_normal_(param.data)
            else:
                torch.nn.init.constant_(param.data, 0)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0
