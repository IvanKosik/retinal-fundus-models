from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform

from bsmu.retinal_fundus.models.utils import debug as debug_utils
from bsmu.retinal_fundus.models.utils import image as image_utils
from bsmu.retinal_fundus.models.utils import train as train_utils


def test_interpolation_quality_during_augmentations(model_trainer):
    image = skimage.io.imread(str(r'C:\MyDiskBackup\Projects\retinal-fundus-models\data\images\hrf_01_dr.png'))
    image = skimage.transform.resize(
        image, model_trainer.config.src_image_shape(), order=3, anti_aliasing=True)  # preserve_range=True)
    image = image_utils.normalized_image(image).astype(np.float32)
    mask = skimage.io.imread(str(r'C:\MyDiskBackup\Projects\retinal-fundus-models\data\masks\hrf_01_dr.png'))
    mask = skimage.transform.resize(
        mask, model_trainer.config.src_mask_shape(), order=3, anti_aliasing=True)  # preserve_range=True)
    mask = image_utils.normalized_image(mask).astype(np.float32)
    image, mask = train_utils.augmentate_image_mask(image, mask, model_trainer.config.AUGMENTATION_TRANSFORMS)

    # plt.imshow(image)
    # plt.show()
    # mask = image_utils.normalized_image(mask).astype(np.float32)

    print(debug_utils.print_info(mask, 'mask'))
    print(np.unique(mask).shape)
    plt.imshow(mask)#, cmap='magma')
    plt.show()

    # exit()

    # rotated_mask = scipy.ndimage.rotate(mask, angle=20, reshape=False)
    # print(debug_utils.print_info(rotated_mask, 'mask scipy'))
    # rotated_mask = image_utils.normalized_image(rotated_mask).astype(np.float32)
    # print(debug_utils.print_info(rotated_mask, 'mask scipy'))
    # print(np.unique(rotated_mask).shape)
    # plt.imshow(rotated_mask, cmap='gray')
    # plt.show()


def create_test_data_with_zeros_masks():
    for image_path in Path(r'D:\Projects\retinal-fundus-models\databases\NoMasks_OnlyForVisualTesting\goodQuality').iterdir():
        if image_path.is_dir():
            continue

        image = skimage.io.imread(str(image_path))
        image = skimage.transform.resize(
            image, (image.shape[0] // 5, image.shape[1] // 5, image.shape[2]), order=3, anti_aliasing=True)
        skimage.io.imsave(str(Path(r'D:\Projects\retinal-fundus-models\databases\NoMasks_OnlyForVisualTesting\goodQuality\png') / f'{image_path.stem}.png'), image)

        empty_mask = np.zeros_like(image)
        skimage.io.imsave(str(Path(
            r'D:\Projects\retinal-fundus-models\databases\NoMasks_OnlyForVisualTesting\goodQuality\empty_masks') / f'{image_path.stem}.png'),
                          empty_mask)
