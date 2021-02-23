import random

import cv2
from albumentations import Resize, DualTransform, ShiftScaleRotate, OpticalDistortion, GridDistortion
from albumentations.augmentations import functional as F
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


class ShiftScaleRotateLinearMask(ShiftScaleRotate):
    def __init__(
            self,
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            value=None,
            mask_value=None,
            shift_limit_x=None,
            shift_limit_y=None,
            always_apply=False,
            p=0.5,
    ):
        super().__init__(shift_limit, scale_limit, rotate_limit, interpolation, border_mode, value,
                         mask_value, shift_limit_x, shift_limit_y, always_apply, p)

    def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, cv2.INTER_LINEAR, self.border_mode, self.mask_value)


class OpticalDistortionLinearMask(OpticalDistortion):
    def __init__(self,
                 distort_limit=0.05,
                 shift_limit=0.05,
                 interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101,
                 value=None,
                 mask_value=None,
                 always_apply=False,
                 p=0.5,
                 ):
        super().__init__(distort_limit, shift_limit, interpolation, border_mode, value, mask_value, always_apply, p)

    def apply_to_mask(self, img, k=0, dx=0, dy=0, **params):
        return F.optical_distortion(img, k, dx, dy, cv2.INTER_LINEAR, self.border_mode, self.mask_value)


class GridDistortionLinearMask(GridDistortion):
    def __init__(self,
                 num_steps=5,
                 distort_limit=0.3,
                 interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101,
                 value=None,
                 mask_value=None,
                 always_apply=False,
                 p=0.5,
                 ):
        super().__init__(num_steps, distort_limit, interpolation, border_mode, value, mask_value, always_apply, p)

    def apply_to_mask(self, img, stepsx=(), stepsy=(), **params):
        return F.grid_distortion(
            img, self.num_steps, stepsx, stepsy, cv2.INTER_LINEAR, self.border_mode, self.mask_value
        )
