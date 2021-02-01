from bsmu.retinal_fundus.models.unet import trainer
# from bsmu.retinal_fundus.models.utils import csv as csv_utils
# from pathlib import Path
# import skimage.io
# import skimage.transform
# import numpy as np
from bsmu.retinal_fundus.models.utils import view as view_utils
from bsmu.retinal_fundus.models.utils import debug as debug_utils
from bsmu.retinal_fundus.models.utils import image as image_utils
import skimage.io


def main():
    # image = skimage.io.imread(str(r'C:\MyDiskBackup\Projects\retinal-fundus-models\data\images\chasedb1_01_l.png'))
    # mask = skimage.io.imread(str(r'C:\MyDiskBackup\Projects\retinal-fundus-models\data\masks\chasedb1_01_l.png'))
    # view = view_utils.ImageMaskView(image, mask, 0.5)
    # view.show()
    # exit()

    # image = skimage.io.imread(str(r'C:\MyDiskBackup\Projects\retinal-fundus-models\data\images\chasedb1_01_l.png'))
    # mask = skimage.io.imread(str(r'C:\MyDiskBackup\Projects\retinal-fundus-models\data\masks\chasedb1_01_l.png'))
    # debug_utils.print_info(image, 'image')
    # debug_utils.print_info(mask, 'mask')
    # view = view_utils.ImageMaskGridView([image] * 4, [mask] * 4, 0.5)
    # view.show()
    # exit()


    print('Run, retinal-fundus-models, run!')

    model_trainer = trainer.UnetModelTrainer()

    # model_trainer.predict_using_generator(model_trainer.test_generator, 1)


    image = skimage.io.imread(str(r'D:\Projects\retinal-fundus-models\databases\HRF_all_GoodImages\images\15_h.jpg'))
    # model_trainer.predict_on_images(images=[image])
    image_tiles = image_utils.split_image_into_tiles(image, (3, 3))
    tile_masks = model_trainer.predict_on_images(images=image_tiles, save=False)
    mask = image_utils.merge_tiles_into_image(tile_masks, (3, 3))

    model_trainer.save_predictions([mask], prefix='combined_mask')


    # model_trainer.verify_generator(model_trainer.train_generator, show=True)

    # csv_utils.generate_train_valid_csv(
    #     model_trainer.config.image_dir(), model_trainer.config.mask_dir(),
    #     model_trainer.config.train_data_csv_path(), model_trainer.config.valid_data_csv_path())

    # model_trainer.run()


if __name__ == '__main__':
    main()


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
