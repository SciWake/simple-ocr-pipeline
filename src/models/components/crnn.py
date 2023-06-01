from typing import List, Optional, Tuple, Union
import json
import torch
from torch import nn


class CRNN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        vocab_path: str,
        input_size: Tuple[int, int, int] = (1024, 64, 3),
        rnn_size: int = 512,
        rnn_num_layers: int = 2,
        rnn_bidirectional: bool = True,
        rnn_dir_merge_type: str = 'cat',
        body_linears: Optional[Tuple[int]] = None,
        type_head: bool = False,
        duet_mode: bool = False,
    ):
        super().__init__()
        self.input_size = input_size

        with open(vocab_path) as file:
            vocab = json.load(file)
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)

        self.rnn_size = rnn_size
        self.type_head = type_head
        self.rnn_num_layers = rnn_num_layers
        self.rnn_num_directions = 2 if rnn_bidirectional else 1
        self.rnn_dir_merge_type = rnn_dir_merge_type
        self.duet_mode = duet_mode

        # cnn часть
        self.cnn = encoder

        for param in self.cnn.parameters():
            param.requires_grad = True

        with torch.no_grad():
            cnn_out_size = self.cnn(torch.rand(1, input_size[2], input_size[1], input_size[0])).shape

        self.post_cnn_flat_size = cnn_out_size[1] * cnn_out_size[2]

        # fcnn между cnn и rnn частью
        linear_in_size = self.post_cnn_flat_size
        self.body_fcnn = []
        for body_linear_size in body_linears or []:
            self.body_fcnn.extend([
                nn.Linear(linear_in_size, body_linear_size),
                nn.ReLU()
            ])
            linear_in_size = body_linear_size
        self.body_fcnn = nn.Sequential(*self.body_fcnn) if self.body_fcnn else nn.Identity()

        self.softmax = nn.LogSoftmax(dim=2)

        self.rnn = decoder

        self.register_backward_hook(self.backward_hook)

    def forward(self, image):
        x = self.cnn(image)  # [b, c, h, w]

        x = x.permute(3, 0, 2, 1)  # [w, b, h, c]
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # [w, b, h*c]

        x = self.body_fcnn(x)  # = nn.Identity [128, b, 2048]

        x = self.rnn(x)  # w, b, features

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

    @torch.jit.export
    def get_input_size(self):
        return self.input_size

    @torch.jit.export
    def get_vocab(self):
        return self.vocab
