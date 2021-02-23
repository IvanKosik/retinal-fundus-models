from __future__ import annotations

import math

import keras
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform

from bsmu.retinal_fundus.models.utils import debug as debug_utils
from bsmu.retinal_fundus.models.utils import image as image_utils


class DataGenerator(keras.utils.Sequence):
    def __init__(self, config: ModelTrainerConfig, data_csv_path: Path, shuffle: bool, augmentation_transforms,
                 discard_last_incomplete_batch: bool = True, tile_grid_shape=None):
        self.config = config
        self.shuffle = shuffle
        self.augmentation_transforms = augmentation_transforms
        self.discard_last_incomplete_batch = discard_last_incomplete_batch
        self.tile_grid_shape = tile_grid_shape

        data_frame = pd.read_csv(str(data_csv_path))
        data = data_frame.to_numpy()
        self.sample_qty = len(data)

        self.images, self.masks = self.read_data(data)

        debug_utils.print_info(self.images, 'images')
        debug_utils.print_info(self.masks, 'masks')

        self.sample_indexes = np.arange(self.sample_qty)
        self.on_epoch_end()

    def read_data(self, data):
        if self.tile_grid_shape is not None:
            return self.read_tiled_data(data)

        images = np.empty(
            shape=(self.sample_qty, *self.config.src_image_shape()), dtype=np.float32)
        masks = np.empty(
            shape=(self.sample_qty, *self.config.src_mask_shape()), dtype=np.float32)

        for index, data_row in enumerate(data):
            image_id = data_row[0]
            print(f'#{index + 1}/{self.sample_qty} \timage_id: {image_id}')

            image_path = self.config.image_dir() / image_id
            image = skimage.io.imread(str(image_path))
            image = skimage.transform.resize(
                image, images.shape[1:], order=3, anti_aliasing=True)  # preserve_range=True)
            image = image_utils.normalized_image(image).astype(np.float32)
            images[index] = image

            mask_path = self.config.mask_dir() / image_id
            mask = skimage.io.imread(str(mask_path))
            mask = skimage.transform.resize(
                mask, masks.shape[1:], order=3, anti_aliasing=True)  # preserve_range=True)
            mask = image_utils.normalized_image(mask).astype(np.float32)
            masks[index] = mask

        return images, masks

    def read_tiled_data(self, data):
        grid_tile_qty = self.tile_grid_shape[0] * self.tile_grid_shape[1]
        self.sample_qty *= grid_tile_qty
        #% self.sample_qty = self.sample_qty + (self.sample_qty * grid_tile_qty)

        images = np.empty(
            shape=(self.sample_qty, *self.config.model_input_image_shape()), dtype=np.float32)
        masks = np.empty(
            shape=(self.sample_qty, *self.config.mask_shape()), dtype=np.float32)

        for index, data_row in enumerate(data):
            image_id = data_row[0]
            print(f'#{index + 1}/{self.sample_qty} \timage_id: {image_id}')

            image_path = self.config.image_dir() / image_id
            image = skimage.io.imread(str(image_path))
            image = skimage.transform.resize(
                image, self.config.src_image_shape(), order=3, anti_aliasing=True)  # preserve_range=True)
            image = image_utils.normalized_image(image).astype(np.float32)
            image_tiles = image_utils.split_image_into_tiles(image, self.tile_grid_shape)
            for tile_index, image_tile in enumerate(image_tiles):
                image_tile = image_utils.normalized_image(image_tile).astype(np.float32)
                images[index * grid_tile_qty + tile_index] = image_tile

            # image = skimage.transform.resize(
            #     image, images.shape[1:], order=3, anti_aliasing=True)  # preserve_range=True)
            # image = image_utils.normalized_image(image).astype(np.float32)
            # images[index * (grid_tile_qty + 1) + grid_tile_qty] = image

            mask_path = self.config.mask_dir() / image_id
            mask = skimage.io.imread(str(mask_path))
            mask = skimage.transform.resize(
                mask, self.config.src_mask_shape(), order=3, anti_aliasing=True)  # preserve_range=True)
            mask = image_utils.normalized_image(mask).astype(np.float32)
            mask_tiles = image_utils.split_image_into_tiles(mask, self.tile_grid_shape)
            for tile_index, mask_tile in enumerate(mask_tiles):
                mask_tile = image_utils.normalized_image(mask_tile).astype(np.float32)
                masks[index * grid_tile_qty + tile_index] = mask_tile

            # mask = skimage.transform.resize(
            #     mask, masks.shape[1:], order=3, anti_aliasing=True)  # preserve_range=True)
            # mask = image_utils.normalized_image(mask).astype(np.float32)
            # masks[index * (grid_tile_qty + 1) + grid_tile_qty] = mask

        return images, masks

    def __len__(self):
        """Return number of batches per epoch"""
        batch_qty = self.sample_qty / self.config.BATCH_SIZE
        return math.floor(batch_qty) if self.discard_last_incomplete_batch else math.ceil(batch_qty)

    def __getitem__(self, batch_index):
        """Generate one batch of data"""
        batch_images = np.zeros(shape=self.config.model_input_batch_shape(), dtype=np.float32)
        batch_masks = np.zeros(shape=self.config.mask_batch_shape(), dtype=np.float32)

        # Generate image indexes of the batch
        batch_sample_indexes = self.sample_indexes[batch_index * self.config.BATCH_SIZE:
                                                   (batch_index + 1) * self.config.BATCH_SIZE]

        for item_number, batch_sample_index in enumerate(batch_sample_indexes):
            image = self.images[batch_sample_index]
            mask = self.masks[batch_sample_index]

            if self.augmentation_transforms is not None:
                image, mask = augmentate_image_mask(image, mask, self.augmentation_transforms)

                # Normalize once again image to [0, 1] after augmentation
                image = image_utils.normalized_image(image)
                mask = image_utils.normalized_image(mask)

            mask = np.round(mask)

            image = image * 255
            batch_images[item_number, ...] = image

            batch_masks[item_number, ...] = mask

        if self.config.PREPROCESS_BATCH_IMAGES is not None:
            batch_images = self.config.PREPROCESS_BATCH_IMAGES(batch_images)

        return batch_images, batch_masks

    def on_epoch_end(self):
        """Shuffle files after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.sample_indexes)


def augmentate_image_mask(image, mask, augmentation_transforms):
    augmentation_results = augmentation_transforms(image=image, mask=mask)
    return augmentation_results['image'], augmentation_results['mask']
