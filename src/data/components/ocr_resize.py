import cv2
import albumentations as A


class OCRAdaptiveResize:
    def __init__(self, height: int, width: int):
        self.h = height
        self.w = width
        self.padding = A.PadIfNeeded(
            self.h, self.w, position='random',
            border_mode=cv2.BORDER_CONSTANT, value=0,
        )

    def __call__(self, force_apply=False, **kwargs):
        image = kwargs['image']
        factor = self.h / image.shape[0]
        w = min(int(factor * image.shape[1]), self.w)
        image = cv2.resize(image, (w, self.h))
        kwargs['image'] = self.padding(image=image)['image']

        return kwargs
