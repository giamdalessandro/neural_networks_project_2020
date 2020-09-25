import os
import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib import colors as clr
import matplotlib.image as mpimg
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

from dataset_utils import load_test_image, FILE_ID


scale1 = 400     # beta * 100
scale2 = 2*scale1*1000  



def compute_heatmap(x, mode="sum"):
    """
    mode can be "sum" or "avg"
    """
    if mode=="sum":
            return x.sum(axis=3, dtype='float32')
    else:
            return x.mean(axis=3, dtype='float32')

def print_heatmap(a, cmap="bone"):
    """
    Prints the heatmap of the raw and the masked feature maps at the same time for an easy comparison
    """

    fig = plt.figure()
    fig.suptitle('Heatmaps of preprocessed image'+FILE_ID)

    ax = []                     # ax enables access to manipulate each of subplots
    images = []                 # aux array to calculate min & max value for the color scale
    cols = len(a) + 1           # min: 3, max: 4

    ax.append(fig.add_subplot(1, cols,1))
    ax[-1].set_title("preprocessed img")
    images.append(plt.imshow(load_test_image()[0, :, :, :], cmap))
    ax[-1].label_outer()

    ax.append(fig.add_subplot(1, cols, 2))
    ax[-1].set_title("conv1 heatmap")
    images.append(plt.imshow(a[0][0, :, :], cmap))
    ax[-1].label_outer()

    ax.append(fig.add_subplot(1, cols, 3))
    ax[-1].set_title("masked1 heatmap")
    images.append(plt.imshow(a[1][0, :, :], cmap))
    ax[-1].label_outer()

    if cols > 3:
        ax.append(fig.add_subplot(1, cols, 4))
        ax[-1].set_title("conv2 heatmap")
        images.append(plt.imshow(a[2][0, :, :], cmap))
        ax[-1].label_outer()

        ax.append(fig.add_subplot(1, cols, 5))
        ax[-1].set_title("masked2 heatmap")
        images.append(plt.imshow(a[3][0, :, :], cmap))
        ax[-1].label_outer()

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)

    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=ax, orientation='horizontal', fraction=.1)

    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacksSM.connect('changed', update)

    plt.show()


def print_feature_maps(x, title, n_imgs=4, cmap="bone"):
    """
    Prints the `(2n_imgs)^2` feature maps at the same time for an easy comparison
    """

    rows = n_imgs*2
    cols = n_imgs*2

    fig = plt.figure()
    fig.suptitle(title+' feature map of '+FILE_ID)

    ax = []                     # ax enables access to manipulate each of subplots
    images = []                 # aux array to calculate min & max value for the color scale

    for i in range(cols*rows):
        ax.append(fig.add_subplot(rows, cols, i+1))   
        ax[-1].set_title(title+" x: " + str(int(i)))
        images.append(plt.imshow(x[0, :, :, i], cmap))      
        ax[i].label_outer()

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)

    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=ax, orientation='horizontal', fraction=.1)

    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacksSM.connect('changed', update)

    plt.show()


def print_comparison2_step(raw_x, masked_x, n_imgs=4, cmap="bone", i=0):
    """
    Prints the `i-th` raw and masked feature maps at the same time for an easy comparison
    """
    ax = []                     # ax enables access to manipulate each of subplots
    images = []                 # aux array to calculate min & max value for the color scale

    fig = plt.figure()
    fig.suptitle('Raw feature maps | Masked(1) feature map | Masked(2) feature map')

    ax.append(fig.add_subplot(1, 2, 1))
    ax[-1].set_title("raw x: " + str(int(i)))
    images.append(plt.imshow(raw_x[0,:,:,i], cmap))
    ax[-1].label_outer()

    ax.append(fig.add_subplot(1, 2, 2))
    ax[-1].set_title("masked x: " + str(int(i)))
    images.append(plt.imshow(masked_x[0, :,:,i], cmap))
    ax[-1].label_outer()

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)

    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[-1], ax=ax, orientation='horizontal', fraction=.1)

    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacksSM.connect('changed', update)

    plt.show()


def print_comparison_step(x, n_imgs=4, cmap="bone", i=0):
    """
    Prints the `i-th` raw and masked feature maps at the same time for an easy comparison
    """
    ax = []                     # ax enables access to manipulate each of subplots
    images = []                 # aux array to calculate min & max value for the color scale

    fig = plt.figure()
    cols = len(x)

    ax.append(fig.add_subplot(1, cols, 1))
    ax[-1].set_title("Conv(1): " + str(int(i)))
    images.append(plt.imshow(x[0][0, :, :, i], cmap))
    ax[-1].label_outer()

    ax.append(fig.add_subplot(1, cols, 2))
    ax[-1].set_title("Masked(1) x: " + str(int(i)))
    images.append(plt.imshow(x[1][0, :, :, i], cmap))
    ax[-1].label_outer()

    if cols > 2:
        ax.append(fig.add_subplot(1, cols, 3))
        ax[-1].set_title("Conv(2) x: " + str(int(i)))
        images.append(plt.imshow(x[2][0, :, :, i], cmap))
        ax[-1].label_outer()

        ax.append(fig.add_subplot(1, cols, 4))
        ax[-1].set_title("Masked(2) x: " + str(int(i)))
        images.append(plt.imshow(x[3][0, :, :, i], cmap))
        ax[-1].label_outer()

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)

    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[-1], ax=ax, orientation='horizontal', fraction=.1)

    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacksSM.connect('changed', update)

    plt.show()

def print_comparison(raw_x, masked_x, n_imgs=4, cmap="bone", step=False):
    """
    Prints `(2n_imgs)^2` raw and masked feature maps for a comparison
    """
    rows = n_imgs*2
    cols = n_imgs*2

    fig = plt.figure()
    fig.suptitle('Raw feature maps | Masked feature map')

    ax = []                     # ax enables access to manipulate each of subplots
    images = []                 # aux array to calculate min & max value for the color scale

    for i in range(cols*rows):
        ax.append(fig.add_subplot(rows, cols, i+1))
        if i%2 == 0:                                            # raw feature maps are on even indeces
            ax[-1].set_title("raw x: " + str(int(((i/2)+1))))
            images.append(plt.imshow(raw_x[0,:,:,i], cmap))
        else:
            ax[-1].set_title("masked x: " + str(int(((i-1)/2)+1)))
            images.append(plt.imshow(masked_x[0,:,:,i-1], cmap))   # i -1 perchè è un for perverso
        
        ax[i].label_outer()

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)

    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=ax, orientation='horizontal', fraction=.1)

    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacksSM.connect('changed', update)

    plt.show()

''' 
code for printing with the same scale based on:

https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/multi_image.html#sphx-glr-gallery-images-contours-and-fields-multi-image-py

https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645
'''


def pretty_little_flower(history):
    #PRINT LOSS FUNCTION HISTORY during EPOCHS
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    #PRINT ACCURACY FUNCTION during EPOCHS
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
