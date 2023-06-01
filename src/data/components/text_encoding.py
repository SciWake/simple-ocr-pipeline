from typing import Dict, List, Union
import numpy as np
import torch


class TextEncoder:
    """
    Кодирует исходный текст.
    """
    def __init__(self, vocab: Union[str, List[str]], target_text_size: int):
        self.vocab = vocab if isinstance(vocab, list) else list(vocab)
        self.target_text_size = target_text_size
        self.blank = 0

    def __call__(self, force_apply=False, **kwargs) -> Dict[str, np.ndarray]:
        source_text = kwargs['text']
        postprocessed_text = []
        last_index = -1
        for char in source_text:
            char_index = self.vocab.index(char) + 1
            if char_index == last_index:
                postprocessed_text.append(self.blank)
            postprocessed_text.append(char_index)
            last_index = char_index

        postprocessed_text = np.pad(
            postprocessed_text,
            (0, self.target_text_size - len(postprocessed_text)),
            mode='constant',
        )
        postprocessed_text = torch.IntTensor(postprocessed_text)

        kwargs['text'] = postprocessed_text

        return kwargs

    def _to_dict(self):
        return {
            '__class_fullname__': 'core.components.transforms.TextEncode',
            'always_apply': False,
            'p': 1.0,
            'vocab': self.vocab,
            'target_text_size': self.target_text_size,
        }
