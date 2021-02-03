import keras.backend
import keras.callbacks


class IncreaseLrAfterEveryBatch(keras.callbacks.Callback):
    def __init__(self, exp: float = 0.95):
        super().__init__()

        self.exp = exp

    def on_train_batch_end(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        lr = pow(lr, self.exp)
        # Set the value back to the optimizer
        keras.backend.set_value(self.model.optimizer.lr, lr)
        print("\nEnd of batch %03d: Learning rate is %f." % (batch, lr))

        assert not (logs is None or 'lr' in logs), 'Cannot log LR'
        logs['lr'] = lr
