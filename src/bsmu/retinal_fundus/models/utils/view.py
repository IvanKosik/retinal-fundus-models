import math

import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class ImageMaskView:
    def __init__(self, image, mask, mask_alpha: float = 0.7):
        self.image = image
        self.mask = mask
        self.mask_alpha = mask_alpha

        self.show_mask = True

        self.fig = plt.figure()
        self.plot1 = self.fig.add_subplot(1, 2, 1)
        self.plot2 = self.fig.add_subplot(1, 2, 2)

        button_axes = self.fig.add_axes([0.7, 0.05, 0.2, 0.075])
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


class ImageMaskGridView:
    def __init__(self, images, masks, mask_alpha: float = 0.7):
        self.images = images
        self.masks = masks
        self.mask_alpha = mask_alpha

        self.grid_side_size = math.ceil(math.sqrt(len(images)))
        self.show_mask = True
        # self.plots = np.empty(shape=(self.grid_side_size, self.grid_side_size), dtype=object)
        self.plots_list = []

        self.fig = plt.figure()
        for x in range(self.grid_side_size):
            for y in range(self.grid_side_size):
                plot = self.fig.add_subplot(self.grid_side_size, self.grid_side_size, x * self.grid_side_size + y + 1)
                # self.plots[x, y] = plot
                self.plots_list.append(plot)

        button_axes = self.fig.add_axes([0.7, 0.02, 0.2, 0.075])
        self.button = Button(button_axes, 'Show/Hide Mask')
        self.button.on_clicked(self.show_hide_mask)

    def show(self):
        self.update()
        plt.show()

    def show_hide_mask(self, event):
        self.show_mask = not self.show_mask
        self.update()

    def update(self):
        for i, image in enumerate(self.images):
            mask = self.masks[i]

            self.plots_list[i].imshow(image, 'gray', interpolation='none')
            if self.show_mask:
                self.plots_list[i].imshow(mask, 'jet', interpolation='none', alpha=self.mask_alpha)

        plt.draw()
