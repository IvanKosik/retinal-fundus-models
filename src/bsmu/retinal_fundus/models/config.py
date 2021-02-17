import typing
from pathlib import Path

import keras.callbacks
from segmentation_models import losses as sm_losses
from segmentation_models import metrics as sm_metrics
from bsmu.retinal_fundus.models.callbacks import IncreaseLrAfterEveryBatch


class ModelTrainerConfig:
    PROJECT_DIR = Path(__file__).parents[4].resolve()
    DATA_DIR = PROJECT_DIR / 'data'
    OUTPUT_DIR = PROJECT_DIR / 'output'

    MODEL_ARCHITECTURE = None
    BACKBONE: str = ''
    PREPROCESS_BATCH_IMAGES: typing.Callable = None

    TRAIN_TILE_GRID_SHAPE = None
    VALID_TILE_GRID_SHAPE = None

    BATCH_SIZE: int = 16
    SRC_IMAGE_SIZE: tuple = (1056, 1056)
    MODEL_INPUT_IMAGE_SIZE: tuple = (500, 500)
    INPUT_CHANNELS: int = 3

    CLASSES_QTY: int = 1

    LR: float = 5e-3  # 1.3e-3
    LOSS = sm_losses.bce_jaccard_loss
    EPOCHS: int = 150

    # For callbacks
    MONITORED_QUANTITY_NAME = 'val_' + sm_metrics.iou_score.name
    MONITORED_QUANTITY_MODE = 'max'

    MODEL_NAME_PREFIX = ''
    MODEL_NAME_POSTFIX = ''

    AUGMENTATION_TRANSFORMS = None

    CSV_TITLE = 'all'
    TRAIN_CSV_NAME = 'train.csv'
    VALID_CSV_NAME = 'valid.csv'
    TEST_CSV_NAME = 'test.csv'

    FIND_LR_START_LR: float = 1e-10
    FIND_LR_EPOCHS: int = 20

    @classmethod
    def model_name(cls):
        size = f'{cls.MODEL_INPUT_IMAGE_SIZE[0]}x{cls.MODEL_INPUT_IMAGE_SIZE[1]}'
        batch_size = f'b{cls.BATCH_SIZE}'
        name_parts = [cls.MODEL_NAME_PREFIX, size, batch_size, cls.MODEL_NAME_POSTFIX]
        name = '_'.join(filter(None, name_parts))
        return f'{name}.h5'

    @classmethod
    def src_image_shape(cls):
        return (*cls.SRC_IMAGE_SIZE, cls.INPUT_CHANNELS)

    @classmethod
    def model_input_image_shape(cls):
        return (*cls.MODEL_INPUT_IMAGE_SIZE, cls.INPUT_CHANNELS)

    @classmethod
    def model_input_batch_shape(cls):
        return (cls.BATCH_SIZE, *cls.model_input_image_shape())

    @classmethod
    def src_mask_shape(cls):
        return (*cls.SRC_IMAGE_SIZE, cls.CLASSES_QTY)

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
    def model_path(cls):
        return cls.model_dir() / cls.model_name()

    @classmethod
    def log_dir(cls):
        return cls.OUTPUT_DIR / 'logs'

    @classmethod
    def log_path(cls):
        return cls.log_dir() / cls.model_name()

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
    def images_csv_dir(cls):
        return cls.csv_dir() / cls.CSV_TITLE

    @classmethod
    def train_data_csv_path(cls):
        return cls.images_csv_dir() / cls.TRAIN_CSV_NAME

    @classmethod
    def valid_data_csv_path(cls):
        return cls.images_csv_dir() / cls.VALID_CSV_NAME

    @classmethod
    def test_data_csv_path(cls):
        return cls.images_csv_dir() / cls.TEST_CSV_NAME

    @classmethod
    def checkpoint_callback(cls):
        return keras.callbacks.ModelCheckpoint(
            str(cls.model_path()), monitor=cls.MONITORED_QUANTITY_NAME, verbose=1,
            save_best_only=True, mode=cls.MONITORED_QUANTITY_MODE)

    @classmethod
    def reduce_lr_callback(cls):
        return keras.callbacks.ReduceLROnPlateau(
            monitor=cls.MONITORED_QUANTITY_NAME, factor=0.8, patience=20, verbose=1,
            mode=cls.MONITORED_QUANTITY_MODE, min_lr=1e-6)

    @classmethod
    def early_stopping_callback(cls):
        return keras.callbacks.EarlyStopping(
            monitor=cls.MONITORED_QUANTITY_NAME, patience=200, mode=cls.MONITORED_QUANTITY_MODE)

    @classmethod
    def tensorboard_callback(cls):
        return keras.callbacks.TensorBoard(log_dir=str(cls.log_path()), write_graph=False)

    @classmethod
    def callbacks(cls):
        return [cls.checkpoint_callback(), cls.reduce_lr_callback(),
                cls.early_stopping_callback(), cls.tensorboard_callback()]

    @classmethod
    def find_lr_callbacks(cls):
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=str(cls.log_path()), write_grads=False, update_freq='batch')
        increase_lr_after_every_batch_callback = IncreaseLrAfterEveryBatch(exp=0.95)

        return [increase_lr_after_every_batch_callback, tensorboard_callback]
