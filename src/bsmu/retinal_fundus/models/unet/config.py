from pathlib import Path

import albumentations
import cv2
from keras.applications import densenet
from segmentation_models import Unet

from bsmu.retinal_fundus.models.augmentations import \
    ElasticSize, ShiftScaleRotateLinearMask, OpticalDistortionLinearMask, GridDistortionLinearMask
from bsmu.retinal_fundus.models.config import ModelTrainerConfig


class UnetModelTrainerConfig(ModelTrainerConfig):
    DATA_DIR = Path(r'C:\MyDiskBackup\Projects\retinal-fundus-models\data')

    MODEL_ARCHITECTURE = Unet
    BACKBONE = 'inceptionv3'
    PREPROCESS_BATCH_IMAGES = densenet.preprocess_input

    # TRAIN_TILE_GRID_SHAPE = (2, 2)
    VALID_TILE_GRID_SHAPE = (2, 2)

    BATCH_SIZE = 8
    SRC_IMAGE_SIZE = (704, 704)
    # SRC_IMAGE_SIZE = (352, 352)
    MODEL_INPUT_IMAGE_SIZE = (352, 352)

    LR = 5e-3
    EPOCHS = 700

    MODEL_NUMBER = 73
    MODEL_NAME_PREFIX = 'InceptionV3'
    MODEL_NAME_POSTFIX = 'Cropped_TiledValid_RoundMask'

    CSV_TITLE = 'chasedb-drive-hrf'

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

        # albumentations.RandomRain(rain_type='drizzle', drop_width=3, blur_value=1, brightness_coefficient=1, p=0.25),

        albumentations.ColorJitter(brightness=0, contrast=0, hue=0.05, p=0.5),

        albumentations.OneOf([
            albumentations.RandomSizedCrop(
                min_max_height=(MODEL_INPUT_IMAGE_SIZE[0], MODEL_INPUT_IMAGE_SIZE[0]),
                height=MODEL_INPUT_IMAGE_SIZE[0], width=MODEL_INPUT_IMAGE_SIZE[1],
                interpolation=cv2.INTER_CUBIC, p=0.8),
            albumentations.RandomSizedCrop(
                min_max_height=(MODEL_INPUT_IMAGE_SIZE[0], SRC_IMAGE_SIZE[0]),
                height=MODEL_INPUT_IMAGE_SIZE[0], width=MODEL_INPUT_IMAGE_SIZE[1],
                interpolation=cv2.INTER_CUBIC, p=0.2),
        ], p=1),

        #ShiftScaleRotateLinearMask(
        albumentations.ShiftScaleRotate(
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_CONSTANT, rotate_limit=20, shift_limit=0.15, scale_limit=0.2, p=1),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.RandomRotate90(p=1),
        ElasticSize(p=0.5),

        #OpticalDistortionLinearMask(
        albumentations.OpticalDistortion(
            distort_limit=0.2, border_mode=cv2.BORDER_CONSTANT, p=0.5),

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

        #GridDistortionLinearMask(
        albumentations.GridDistortion(
            num_steps=4, distort_limit=0.15, border_mode=cv2.BORDER_CONSTANT, p=0.75),

    ], p=1.0)
