from pathlib import Path

import albumentations
import cv2
from keras.applications import densenet
from segmentation_models import Unet

from bsmu.retinal_fundus.models.augmentations import ElasticSize
from bsmu.retinal_fundus.models.config import ModelTrainerConfig


class UnetModelTrainerConfig(ModelTrainerConfig):
    DATA_DIR = Path(r'C:\MyDiskBackup\Projects\retinal-fundus-models\data')

    MODEL_ARCHITECTURE = Unet
    BACKBONE = 'densenet201'
    PREPROCESS_BATCH_IMAGES = densenet.preprocess_input

    BATCH_SIZE = 8
    MODEL_INPUT_IMAGE_SIZE = (352, 352)

    MODEL_NAME_PREFIX = 'DenseNet201'
    MODEL_NAME_POSTFIX = 'Test1'

    AUGMENTATION_TRANSFORMS = albumentations.Compose([
        # albumentations.ShiftScaleRotate(
        #     border_mode=cv2.BORDER_CONSTANT, rotate_limit=20, shift_limit=0.2, scale_limit=0.2,
        #     p=1.0),  # TODO try: interpolation=cv2.INTER_CUBIC
        # albumentations.HorizontalFlip(p=0.5),
        #
        # # Additional augmentations
        # albumentations.RandomGamma(p=0.5),
        # albumentations.IAASharpen(p=0.5),
        # albumentations.OpticalDistortion(p=0.5),
        # albumentations.RandomBrightnessContrast(p=0.2)



        albumentations.ShiftScaleRotate(
            border_mode=cv2.BORDER_CONSTANT, rotate_limit=20, shift_limit=0.15, scale_limit=0.2, p=1),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.RandomRotate90(p=1),
        ElasticSize(p=0.5),


        albumentations.OpticalDistortion(distort_limit=0.2, border_mode=cv2.BORDER_CONSTANT, p=0.5),

        albumentations.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.25, p=1),
        albumentations.RandomGamma(p=1),

        albumentations.Compose(transforms=[
            albumentations.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=1),
            albumentations.Blur(blur_limit=3, p=1),
        ], p=0.2),

        albumentations.IAAEmboss(p=0.3),
        albumentations.IAASharpen(alpha=(0, 1), lightness=(5, 10), p=0.15),

        albumentations.OneOf([
            albumentations.Blur(blur_limit=3, p=1),
            albumentations.MotionBlur(blur_limit=3, p=1),
            albumentations.MedianBlur(blur_limit=3, p=1),
        ], p=0.15),


        albumentations.GridDistortion(num_steps=4, distort_limit=0.15, border_mode=cv2.BORDER_CONSTANT, p=0.75),

    ], p=1.0)
