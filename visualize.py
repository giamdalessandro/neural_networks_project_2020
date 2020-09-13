import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

path = '/media/luca/DATA2/uni/neural_networks_project_2020/dataset/detanimalpart/n02355227_obj/img/img/00007.jpg'


def loaddd():
    img = load_img(path, target_size=(224, 224))      # test image
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def compute_heatmap(model, img, mask):
    feature_maps = model.predict(img)
    if mask:
        #temp = feature_maps.sum(axis=2, dtype='float32')        # heatmap = somma ???
        temp = feature_maps.mean(axis=2, dtype='float32')        # heatmap = somma ???

    else:
        #temp = feature_maps.sum(axis=3, dtype='float32')        # heatmap = somma ???
        temp = feature_maps.mean(axis=3, dtype='float32')        # heatmap = somma ???

    return temp


def print_heatmap(img1, img2):

    '''
    fig = plt.figure()
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 3),
                    axes_pad=0.05,
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    ax = plt.subplot(1, 3, 1)
    im = ax.imshow(loaddd()[0, :, :, :])        # preprocessed image
    
    ax = plt.subplot(1, 3, 2)
    im = ax.imshow(img1[0,:,:], cmap='bone')    # feature maps

    ax = plt.subplot(1, 3, 3)
    im = ax.imshow(img2[:,:], cmap='bone')      # masked feature maps

    grid.cbar_axes[0].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(True)

    plt.show()

    '''
    ax = plt.subplot(1, 3, 1)
    ax.set_title('preprocessed image')
    plt.imshow(loaddd()[0, :, :, :])        # preprocessed image
    
    ax = plt.subplot(1, 3, 2)
    ax.set_title('feature maps')
    plt.imshow(img1[0,:,:], cmap='bone')    # feature maps
    plt.colorbar()

    ax = plt.subplot(1, 3, 3)
    ax.set_title('masked feature maps')
    plt.imshow(img2[:,:], cmap='bone')      # masked feature maps
    plt.colorbar()
    
    plt.show()

    
def print_feature_maps(model, masked):
    feature_maps = model.predict(loaddd())
    square = 4
    ix = 1

    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            if masked:
                plt.title("masked feature maps")
                plt.imshow(feature_maps[:, :, ix-1], cmap='bone')
            else:
                plt.title("original feature maps")
                plt.imshow(feature_maps[0, :, :, ix-1], cmap='bone')
            ix += 1
            plt.colorbar()
    # show the figure
    plt.show()
