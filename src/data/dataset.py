from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class OCRDataset(Dataset):
    def __init__(
        self,
        annotation_name: Union[str, pd.DataFrame],
        transform: Optional[Callable],
        cache: bool = False,
        max_samples: Optional[int] = None,
        model_output_length: int = 64,
    ):
        if isinstance(annotation_name, pd.DataFrame):
            self.annotation = annotation_name
        elif isinstance(annotation_name, str):
            self.annotation = pd.read_csv(annotation_name, index_col=0)

        self.transform = transform

        if max_samples is not None:
            self.annotation = self.annotation.iloc[: max_samples]

        self.model_output_length = model_output_length

        if cache:
            self.images = tuple(cv2.imread(image_name)[..., ::-1] for image_name in self.annotation['image'].values)
        else:
            self.images = None

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if self.images is None:
            image = cv2.imread(self.annotation['image'][idx])[..., ::-1]
        else:
            image = self.images[idx]

        text = self.annotation['text'][idx]

        input_length = torch.LongTensor([self.model_output_length])
        data = {'image': image, 'text': text, 'text_length': len(text), 'input_length': input_length}

        if self.transform:
            data = self.transform(**data)

        return data
