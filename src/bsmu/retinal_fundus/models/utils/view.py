import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class ImageMaskView:
    def __init__(self, image, mask, mask_alpha: float = 0.7):
        self.image = image
        self.mask = mask
        self.mask_alpha = mask_alpha

        self.show_mask = True

        self.fig = plt.figure()
        self.plot1 = plt.subplot(1, 2, 1)
        self.plot2 = plt.subplot(1, 2, 2)

        button_axes = plt.axes([0.7, 0.05, 0.2, 0.075])
        self.button = Button(button_axes, 'Show/Hide Mask')
        self.button.on_clicked(self.show_hide_mask)

    def show(self):
        self.update()
        plt.show()

    def show_hide_mask(self, event):
        self.show_mask = not self.show_mask
        self.update()

    def update(self):
        self.plot1.imshow(self.image, 'gray', interpolation='none')
        self.plot2.imshow(self.image, 'gray', interpolation='none')

        if self.show_mask:
            self.plot2.imshow(self.mask, 'jet', interpolation='none', alpha=self.mask_alpha)

        plt.draw()
