from bsmu.retinal_fundus.models import trainer
from bsmu.retinal_fundus.models.unet.config import UnetModelTrainerConfig


class UnetModelTrainer(trainer.ModelTrainer):
    def __init__(self):
        super().__init__(UnetModelTrainerConfig)

    def create_model(self):
        super().create_model()

        self.model = self.config.MODEL_ARCHITECTURE(
            backbone_name=self.config.BACKBONE, input_shape=self.config.model_input_image_shape(),
            classes=self.config.CLASSES_QTY, encoder_weights='imagenet', encoder_freeze=True)

        self.model.summary(line_length=150)
