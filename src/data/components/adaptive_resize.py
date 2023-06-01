from albumentations import Resize
import cv2
import numpy as np


class AdaptiveResize(Resize):
    def __init__(self, height: int, width: int):
        super().__init__(height, width)

    def apply(self, image: np.ndarray, **kwargs):
        kwargs.pop('interpolation')
        method = cv2.INTER_CUBIC

        if image.shape[0] > self.height:
            method = cv2.INTER_LINEAR

        return super().apply(image, interpolation=method, **kwargs)
