import typing
from pathlib import Path

from segmentation_models import losses as sm_losses


class ModelTrainerConfig:
    PROJECT_DIR = Path(__file__).parents[4].resolve()
    DATA_DIR = PROJECT_DIR / 'data'
    OUTPUT_DIR = PROJECT_DIR / 'output'

    MODEL_ARCHITECTURE = None
    BACKBONE: str = ''
    PREPROCESS_BATCH_IMAGES: typing.Callable = None

    BATCH_SIZE: int = 16
    MODEL_INPUT_IMAGE_SIZE = (500, 500)
    INPUT_CHANNELS: int = 3

    CLASSES_QTY: int = 1

    LR: float = 1e-4  # 1.3e-3
    LOSS = sm_losses.bce_jaccard_loss
    EPOCHS: int = 150

    MODEL_NAME_PREFIX = ''
    MODEL_NAME_POSTFIX = ''

    AUGMENTATION_TRANSFORMS = None

    TRAIN_CSV_NAME = 'train.csv'
    VALID_CSV_NAME = 'valid.csv'
    TEST_CSV_NAME = 'test.csv'

    @classmethod
    def model_input_image_shape(cls):
        return (*cls.MODEL_INPUT_IMAGE_SIZE, cls.INPUT_CHANNELS)

    @classmethod
    def model_input_batch_shape(cls):
        return (cls.BATCH_SIZE, *cls.model_input_image_shape())

    @classmethod
    def mask_shape(cls):
        return (*cls.MODEL_INPUT_IMAGE_SIZE, cls.CLASSES_QTY)

    @classmethod
    def mask_batch_shape(cls):
        return (cls.BATCH_SIZE, *cls.mask_shape())

    @classmethod
    def image_dir(cls):
        return cls.DATA_DIR / 'images'

    @classmethod
    def mask_dir(cls):
        return cls.DATA_DIR / 'masks'

    @classmethod
    def model_dir(cls):
        return cls.OUTPUT_DIR / 'models'

    @classmethod
    def log_dir(cls):
        return cls.OUTPUT_DIR / 'logs'

    @classmethod
    def predicts_dir(cls):
        return cls.OUTPUT_DIR / 'predicts'

    @classmethod
    def test_generator_dir(cls):
        return cls.OUTPUT_DIR / 'test_generator'

    @classmethod
    def csv_dir(cls):
        return cls.DATA_DIR / 'csv'

    @classmethod
    def all_images_csv_dir(cls):
        return cls.csv_dir() / 'all'

    @classmethod
    def train_data_csv_path(cls):
        return cls.all_images_csv_dir() / cls.TRAIN_CSV_NAME

    @classmethod
    def valid_data_csv_path(cls):
        return cls.all_images_csv_dir() / cls.VALID_CSV_NAME

    @classmethod
    def test_data_csv_path(cls):
        return cls.all_images_csv_dir() / cls.TEST_CSV_NAME

    @classmethod
    def part_images_csv_dir(cls):
        return cls.csv_dir() / 'part'

    @classmethod
    def part_train_data_csv_path(cls):
        return cls.part_images_csv_dir() / cls.TRAIN_CSV_NAME

    @classmethod
    def part_valid_data_csv_path(cls):
        return cls.part_images_csv_dir() / cls.VALID_CSV_NAME
