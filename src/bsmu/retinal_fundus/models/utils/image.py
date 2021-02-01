from __future__ import annotations

import cv2
import numpy as np
import skimage.io
import math


def normalized_image(image):
    """
    :param image: two- or three-dimensional image
    :return: normalized to [0, 1] image
    """
    image_min = image.min()
    if image_min != 0:
        image = image - image_min
    image_max = image.max()
    if image_max != 0 and image_max != 1:
        image = image / image_max
    return image


def normalized_8U_image(image):
    """
    :param image: two- or three-dimensional image
    :return: normalized to [0, 255] image
    """
    return (normalized_image(image) * 255).astype(np.uint8)


def normalized_8UC3_image(image):
    """
    :param image: two-dimensional image
    :return: normalized to [0, 255] three-dimensional image
    """
    assert len(image.shape) == 2, 'two-dimensional images are only supported'

    image = normalized_8U_image(image)
    return np.stack((image,) * 3, axis=-1)


def crop_important_image_region(image, probabilistic_mask, threshold=0.05, keep_close_aspect_ratio=False):
    # img is 2D image data
    image = cv2.resize(image, (max(image.shape), max(image.shape)))

    image_size = image.shape[:2]
    probabilistic_mask = cv2.resize(probabilistic_mask, image_size, interpolation=cv2.INTER_CUBIC)
    bool_mask = probabilistic_mask > threshold
    mask_cols, mask_rows = bool_mask.any(0), bool_mask.any(1)
    start_col, end_col = start_end_nonzero_indexes(mask_cols)
    start_row, end_row = start_end_nonzero_indexes(mask_rows)

    if keep_close_aspect_ratio:
        # Try to align a little bit width and height of cropped image
        # to keep close aspect ratio
        new_width = end_col - start_col
        new_height = end_row - start_row
        max_side = max(new_width, new_height)
        delta_width = max_side - new_width
        delta_height = max_side - new_height
        half_delta_width = int(round(delta_width / 2))
        half_delta_height = int(round(delta_height / 2))
        start_col -= half_delta_width
        end_col += half_delta_width
        start_row -= half_delta_height
        end_row += half_delta_height
        start_col = max(0, start_col)
        start_row = max(0, start_row)
        end_col = min(end_col, image.shape[1])
        end_row = min(end_row, image.shape[0])

    return image[start_row:end_row, start_col:end_col]


def start_end_nonzero_indexes(mask):
    return mask.argmax(), mask.shape[0] - mask[::-1].argmax()


def split_image_into_tiles(image, tile_shape: tuple = (2, 2)) -> list:
    row_tile_qty = tile_shape[0]
    col_tile_qty = tile_shape[1]
    row_tile_size = int(round(image.shape[0] / row_tile_qty))
    col_tile_size = int(round(image.shape[1] / col_tile_qty))

    tiles = []
    tile_row_begin = 0
    for row in range(row_tile_qty):
        tile_row_end = image.shape[0] if row == row_tile_qty - 1 else tile_row_begin + row_tile_size
        tile_col_begin = 0
        for col in range(col_tile_qty):
            tile_col_end = image.shape[1] if col == col_tile_qty - 1 else tile_col_begin + col_tile_size
            tile = image[tile_row_begin:tile_row_end, tile_col_begin:tile_col_end, ...]
            tiles.append(tile)

            tile_col_begin = tile_col_end

        tile_row_begin = tile_row_end

    return tiles

    # mid_row = image.shape[0] // 2
    # mid_col = image.shape[1] // 2
    #
    # tile1 = image[:mid_row, :mid_col, ...]
    # tile2 = image[:mid_row, mid_col:, ...]
    # tile3 = image[mid_row:, :mid_col, ...]
    # tile4 = image[mid_row:, mid_col:, ...]
    #
    # return [tile1, tile2, tile3, tile4]


def merge_tiles_into_image(tiles, tile_shape: tuple):
    row_qty = tile_shape[0]
    col_qty = tile_shape[1]

    rows = []
    row_begin_tile_index = 0
    for row in range(row_qty):
        row_end_tile_index = row_begin_tile_index + col_qty
        merged_row_tile = np.concatenate(tiles[row_begin_tile_index:row_end_tile_index], axis=1)
        rows.append(merged_row_tile)

        row_begin_tile_index = row_end_tile_index

    return np.concatenate(rows, axis=0)


def convert_images_to_png(src_image_dir: Path, dst_image_dir: Path):
    for image_path in src_image_dir.iterdir():
        if image_path.is_dir():
            continue

        image = skimage.io.imread(str(image_path))
        skimage.io.imsave(str(dst_image_dir / f'{image_path.stem}.png'), image)
