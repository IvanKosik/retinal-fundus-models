import random

import cv2
from albumentations import Resize, DualTransform
from imgaug import augmenters as iaa


class ElasticSize(DualTransform):
    def __init__(self, pad_limit=0.3, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

        self.pad_limit = pad_limit
        # TODO: |self.border_mode| is not used now.
        self.border_mode = border_mode
        self.value = value

    def apply(self, img, pad_side=0, pad_factor=0, **params):
        return self._transformed(img, pad_side, pad_factor)

    def apply_to_mask(self, img, pad_side=0, pad_factor=0, **params):
        return self._transformed(img, pad_side, pad_factor)

    def _transformed(self, img, pad_side=0, pad_factor=0):
        side_percents = [0] * 4
        side_percents[pad_side] = pad_factor

        w, h = img.shape[:2]

        aug = iaa.CropAndPad(percent=tuple(side_percents), pad_cval=self.value)
        img = aug.augment_image(img)

        resize = Resize(w, h)
        return resize.apply(img)

    def get_params(self):
        return {
            'pad_side': random.choice([0, 1, 2, 3]),
            'pad_factor': random.uniform(0, self.pad_limit),  # (-self.pad_limit, self.pad_limit),
        }

    def get_transform_init_args(self):
        return {
            'pad_limit': self.pad_limit,
            'border_mode': self.border_mode,
            'value': self.value,
        }
