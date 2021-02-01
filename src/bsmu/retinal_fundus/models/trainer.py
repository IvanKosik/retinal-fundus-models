from pathlib import Path
from typing import Type

import keras
import numpy as np
import skimage.io
import skimage.transform
from keras import optimizers
from segmentation_models import losses as sm_losses
from segmentation_models import metrics as sm_metrics

from bsmu.retinal_fundus.models.config import ModelTrainerConfig
from bsmu.retinal_fundus.models.utils import image as image_utils, train as train_utils, debug as debug_utils
from bsmu.retinal_fundus.models.utils import view as view_utils


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

    def verify_generator(self, generator, show: bool = True, save_to_disk: bool = False, batch_qty: int = 7):
        debug_utils.print_title(self.verify_generator.__name__)

        generator_len = len(generator)
        print('generator_len (number of batches per epoch):', generator_len)

        for batch_index in range(min(batch_qty, generator_len)):
            batch = generator.__getitem__(batch_index)
            self.verify_batch(batch, str(batch_index), show, save_to_disk)

    def verify_batch(self, batch, batch_prefix: str, show: bool = True, save_to_disk: bool = False):
        batch_images, batch_masks = batch

        debug_utils.print_info(batch_images, 'batch_images')
        debug_utils.print_info(batch_masks, 'batch_masks')

        normalized_batch_images = []

        # Save all batch images, masks
        for batch_sample_index in range(len(batch_images)):
            image = batch_images[batch_sample_index]
            mask = batch_masks[batch_sample_index]

            debug_utils.print_info(image, '\nimage')
            debug_utils.print_info(mask, '\nmask')

            image = image_utils.normalized_image(image)
            normalized_batch_images.append(image)

            if save_to_disk:
                name = f'{batch_prefix}_{batch_sample_index}.png'
                skimage.io.imsave(str(self.config.test_generator_dir() / 'images' / name), image)

                skimage.io.imsave(str(self.config.test_generator_dir() / 'masks' / name), mask)

        if show:
            view = view_utils.ImageMaskGridView(normalized_batch_images, batch_masks[..., 0], mask_alpha=0.5)
            view.show()

    def predict_using_generator(self, generator, batch_qty: int = 2):
        debug_utils.print_title(self.predict_using_generator.__name__)

        self.load_model()

        generator_len = len(generator)
        for batch_index in range(min(batch_qty, generator_len)):
            batch = generator.__getitem__(batch_index)
            self.predict_batch(batch, str(batch_index))

    def predict_batch(self, batch, batch_prefix: str):
        batch_images, _ = batch
        batch_predicted_masks = self.model.predict(batch_images)

        predictions_dir = self.config.predicts_dir() / Path(self.model_name).stem / 'predicted_masks'
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Save all batch images, masks
        for batch_sample_index in range(len(batch_images)):
            image = batch_images[batch_sample_index]
            predicted_mask = batch_predicted_masks[batch_sample_index]

            debug_utils.print_info(image, '\nimage')
            debug_utils.print_info(predicted_mask, '\npredicted_mask')

            #% image = image_utils.normalized_image(image)
            name = f'{batch_prefix}_{batch_sample_index}.png'
            #% skimage.io.imsave(str(self.config.test_generator_dir() / 'images' / name), image)

            skimage.io.imsave(str(predictions_dir / name), predicted_mask)

    def save_predictions(self, mask_batch, prefix: str):
        predictions_dir = self.config.predicts_dir() / Path(self.model_name).stem / 'predicted_masks'
        predictions_dir.mkdir(parents=True, exist_ok=True)

        for index, mask in enumerate(mask_batch):
            debug_utils.print_info(mask, '\npredicted_mask')

            name = f'{prefix}_{index}.png'
            skimage.io.imsave(str(predictions_dir / name), mask)

    def predict_on_images(
            self, images: list, resize_mask_to_image: bool = True,
            save: bool = True, prefix: str = 'predict_on_images'):
        debug_utils.print_title(self.predict_on_images.__name__)

        self.load_model()

        image_batch = self.create_batch_from_images(images)
        predicted_masks = self.model.predict(image_batch)

        if resize_mask_to_image:
            resized_masks = []
            for index, image in enumerate(images):
                mask = predicted_masks[index]
                resized_mask = skimage.transform.resize(mask, image.shape[:2], order=3, anti_aliasing=True)
                resized_masks.append(resized_mask)
            predicted_masks = resized_masks

        if save:
            self.save_predictions(predicted_masks, prefix)

        return predicted_masks

    def create_batch_from_images(self, images: list):
        image_batch = np.empty(shape=(len(images), *self.config.model_input_image_shape()), dtype=np.float32)

        for image_index, image in enumerate(images):
            image = skimage.transform.resize(image, self.config.model_input_image_shape(), order=3, anti_aliasing=True)
            image = image_utils.normalized_image(image)
            image = image * 255
            image_batch[image_index, ...] = image

        if self.config.PREPROCESS_BATCH_IMAGES is not None:
            image_batch = self.config.PREPROCESS_BATCH_IMAGES(image_batch)

        return image_batch

    def test_model(self):
        ...
