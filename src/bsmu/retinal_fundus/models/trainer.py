from typing import Type

import keras
import numpy as np
import pandas as pd
import skimage.io
from keras import optimizers
from segmentation_models import losses as sm_losses
from segmentation_models import metrics as sm_metrics

from bsmu.retinal_fundus.models.config import ModelTrainerConfig
from bsmu.retinal_fundus.models.utils import image as image_utils, train as train_utils, debug as debug_utils


class ModelTrainer:
    def __init__(self, config: Type[ModelTrainerConfig] = ModelTrainerConfig, model_custom_objects=None):
        self.config = config
        self.model_custom_objects = model_custom_objects

        self.model = None

        self._train_generator = None
        self._valid_generator = None
        self._test_generator = None

        self.callbacks = self._create_callbacks()

    @property
    def model_name(self):
        size = f'{self.config.MODEL_INPUT_IMAGE_SIZE[0]}x{self.config.MODEL_INPUT_IMAGE_SIZE[1]}'
        batch_size = f'b{self.config.BATCH_SIZE}'
        name_parts = [self.config.MODEL_NAME_PREFIX, size, batch_size, self.config.MODEL_NAME_POSTFIX]
        name = '_'.join(filter(None, name_parts))
        return f'{name}.h5'

    @property
    def model_path(self):
        return self.config.model_dir() / self.model_name

    @property
    def log_path(self):
        return self.config.log_dir() / self.model_name

    def run(self):
        if self.model_path.exists():
            self.load_model()
        else:
            self.create_model()

        self._freeze_layers()

        self._train_model()

    def create_model(self):
        debug_utils.print_title(self.create_model.__name__)

    def load_model(self):
        debug_utils.print_title(self.load_model.__name__)

        self.model = keras.models.load_model(
            str(self.model_path), custom_objects=self.model_custom_objects, compile=False)

    def _freeze_layers(self):
        debug_utils.print_title(self._freeze_layers.__name__)

    def _unfreeze_all_layers(self):
        for layer in self.model.layers:
            layer.trainable = True

    def _create_callbacks(self):
        monitored_quantity_name = 'val_' + sm_metrics.iou_score.name
        monitored_quantity_mode = 'max'
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            str(self.model_path), monitor=monitored_quantity_name, verbose=1, save_best_only=True,
            mode=monitored_quantity_mode)
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor=monitored_quantity_name, factor=0.8, patience=6, verbose=1, mode=monitored_quantity_mode,
            min_lr=1e-6)
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor=monitored_quantity_name, patience=60, mode=monitored_quantity_mode)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(self.log_path), write_graph=False)

        return [checkpoint_callback, reduce_lr_callback, early_stopping_callback, tensorboard_callback]

    @property
    def train_generator(self):
        if self._train_generator is None:
            self._create_train_generator()
        return self._train_generator

    def _create_train_generator(self):
        self._train_generator = train_utils.DataGenerator(
            config=self.config, data_csv_path=self.config.train_data_csv_path(), shuffle=True,
            augmentation_transforms=self.config.AUGMENTATION_TRANSFORMS)

    @property
    def valid_generator(self):
        if self._valid_generator is None:
            self._create_valid_generator()
        return self._valid_generator

    def _create_valid_generator(self):
        self._valid_generator = train_utils.DataGenerator(
            config=self.config, data_csv_path=self.config.valid_data_csv_path(), shuffle=False,
            augmentation_transforms=None)

    @property
    def test_generator(self):
        if self._test_generator is None:
            self._create_test_generator()
        return self._test_generator

    def _create_test_generator(self):
        self._test_generator = train_utils.DataGenerator(
            config=self.config, data_csv_path=self.config.test_data_csv_path(), shuffle=False,
            augmentation_transforms=None)

    def _train_model(self):
        debug_utils.print_title(self._train_model.__name__)

        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.config.LR), loss=self.config.LOSS,
                           metrics=[sm_losses.binary_crossentropy, sm_losses.JaccardLoss(per_image=True),
                                    sm_metrics.IOUScore(threshold=0.5, per_image=True)])
        self.model.fit_generator(generator=self.train_generator,
                                 epochs=self.config.EPOCHS,
                                 verbose=2,
                                 callbacks=self.callbacks,
                                 validation_data=self.valid_generator)

    def _print_layers_info(self):
        debug_utils.print_title(self._print_layers_info.__name__)
        debug_utils.print_layers_info(self.model)

    def verify_generator(self, generator, number_of_batches: int = 7):
        debug_utils.print_title(self.verify_generator.__name__)

        generator_len = len(generator)
        print('generator_len (number of batches per epoch):', generator_len)

        for batch_index in range(min(number_of_batches, generator_len)):
            batch = generator.__getitem__(batch_index)
            self.verify_batch(batch, str(batch_index))

    def verify_batch(self, batch, batch_prefix: str):
        batch_input, batch_ages = batch
        batch_predictions = None
        batch_images = None
        if self.combined_model:
            batch_males, batch_predictions = batch_input[0], batch_input[1]
        else:
            batch_images, batch_males = batch_input[0], batch_input[1]

        if batch_images is not None:
            debug_utils.print_info(batch_images, 'batch_images')
        debug_utils.print_info(batch_males, 'batch_males')
        debug_utils.print_info(batch_ages, 'batch_ages')
        if batch_predictions is not None:
            debug_utils.print_info(batch_predictions, 'batch_predictions')

        # Save all batch images
        for batch_sample_index in range(len(batch_males)):
            if batch_images is not None:
                image = batch_images[batch_sample_index]       #### [...]
                debug_utils.print_info(image, '\nimage')
                image = image_utils.normalized_image(image)
                skimage.io.imsave(str(self.config.test_generator_dir() / f'{batch_prefix}_{batch_sample_index}.png'), image)

            male = batch_males[batch_sample_index][0]
            age = batch_ages[batch_sample_index][0]

            print('male:', male)
            print('age:', age)
            if batch_predictions is not None:
                print('predictions:', batch_predictions[batch_sample_index])

    def create_batch_from_one_sample(self, image, male: bool):
        assert len(image.shape) == 2, 'one channel images are only supported'
        assert image.shape == self.model_input_image_size, 'image size is not equal to model input size'

        # Create a batch from one image
        batch_images = np.zeros(shape=(self.BATCH_SIZE, *self.MODEL_INPUT_IMAGE_SHAPE), dtype=np.float32)
        batch_males = np.zeros(shape=(self.BATCH_SIZE, 1), dtype=np.uint8)

        image = image_utils.normalized_image(image)
        image = image * 255
        image = np.stack((image,) * self.model_input_image_channels_count, axis=-1)
        batch_images[0, ...] = image

        batch_males[0, 0] = male

        batch_images = self.preprocess_batch_images(batch_images)

        input_batch = [batch_images, batch_males]
        return input_batch

    def generate_image_cam(self, image, male: bool):
        input_batch = self.create_batch_from_one_sample(image, male)
        cam_batch, output_age_batch, image_cam_overlay_batch = self.generate_cam_batch(input_batch)
        return cam_batch[0], train_utils.denormalized_age(output_age_batch[0][0]), \
            None if image_cam_overlay_batch is None else image_cam_overlay_batch[0]

    def generate_cam_batch(self, input_batch):
        assert self.INPUT_IMAGE_LAYER_NAME and self.INPUT_MALE_LAYER_NAME \
               and self.OUTPUT_AGE_LAYER_NAME and self.OUTPUT_CONV_LAYER_NAME and self.OUTPUT_POOLING_LAYER_NAME, \
               'define all needed layer names to generate activation map'
        return cam_utils.generate_cam_batch(
            input_batch, self.model, self.INPUT_IMAGE_LAYER_NAME, self.INPUT_MALE_LAYER_NAME,
            self.OUTPUT_AGE_LAYER_NAME, self.OUTPUT_CONV_LAYER_NAME, self.OUTPUT_POOLING_LAYER_NAME,
            self.OUTPUT_IMAGE_CAM_OVERLAY_LAYER_NAME)

    def generate_image_cam_overlay(self, image, male: bool, cam_threshold):
        cam, age, image_cam_overlay = self.generate_image_cam(image, male)
        overlay_result = cam_utils.overlay_cam(
            image if image_cam_overlay is None else image_cam_overlay, cam, cam_threshold)
        return overlay_result, age

    def crop_image_to_cam(self, image_src, image, male: bool, threshold):
        cam, age, image_cam_overlay = self.generate_image_cam(image, male)
        return image_utils.crop_important_image_region(image_src, cam, threshold)

    def data_frame_with_predictions(self, csv_path):
        csv_data = pd.read_csv(str(csv_path))
        data = csv_data.to_numpy()

        predictions = self.csv_predictions(csv_path)

        data_with_predictions = np.hstack((data, predictions))
        column_labels = np.append(csv_data.columns, self.model_name)
        return pd.DataFrame(data=data_with_predictions, columns=column_labels)

    def csv_predictions(self, csv_path):
        generator = train_utils.DataGenerator(
            self.IMAGE_DIR, csv_path, self.BATCH_SIZE, self.MODEL_INPUT_IMAGE_SHAPE,
            shuffle=False, preprocess_batch_images=self.preprocess_batch_images,
            augmentation_transforms=None, apply_age_normalization=self.apply_age_nomalization,
            combined_model=self.combined_model, discard_last_incomplete_batch=False)

        predictions = self.model.predict_generator(generator=generator)
        # Remove last rows (predictions for black images, which can be,
        # if the number of images is not a multiple of the batch size)
        predictions = predictions[:generator.sample_qty]
        if self.apply_age_nomalization:
            predictions = train_utils.denormalized_age(predictions)
        return predictions

    def test_model(self):
        ...
