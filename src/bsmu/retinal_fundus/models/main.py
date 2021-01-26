from bsmu.retinal_fundus.models.unet import trainer
from bsmu.retinal_fundus.models.utils import csv as csv_utils


def main():
    print('Run, retinal-fundus-models, run!')

    model_trainer = trainer.UnetModelTrainer()

    # model_trainer.verify_generator(model_trainer.train_generator)

    # csv_utils.generate_train_valid_csv(
    #     model_trainer.config.image_dir(), model_trainer.config.mask_dir(),
    #     model_trainer.config.train_data_csv_path(), model_trainer.config.valid_data_csv_path())

    model_trainer.run()


if __name__ == '__main__':
    main()
