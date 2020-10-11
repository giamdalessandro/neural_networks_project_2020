import tensorflow as tf
import matplotlib.image as mpimg

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from matplotlib import pyplot as plt

from classes.maskLayer import MaskLayer

from utils.visuallib import *
from utils.dataset_utils import load_keras


model_raw1 = load_keras()

model_masked1 = load_keras()
model_masked1.add(MaskLayer())
'''
model_raw2 = load_keras()
model_raw2.add(MaskLayer())
model_raw2.add(Conv2D(512, [3, 3], padding="same",
                  activation='relu', name="our_conv"))

model_masked2 = load_keras()
model_masked2.add(MaskLayer())
model_masked2.add(Conv2D(512, [3, 3], padding="same",
                  activation='relu', name="our_conv"))
model_masked2.add(MaskLayer())
'''
POS_IMAGE_SET_TEST = "./dataset/train_val/bird"
raw_x1 = model_raw1.predict(load_test_image(
    POS_IMAGE_SET_TEST, "2008_001679.jpg"))
#raw_x2 = model_raw2.predict(load_test_image(POS_IMAGE_SET_TEST, "2008_000512.jpg"))
#raw_x2 = 1000 * raw_x2

# we use their parameters but scale the masked x before plotting for visual reason
masked_x1 = model_masked1.predict(
    load_test_image(POS_IMAGE_SET_TEST, "2008_001679.jpg"))
masked_x1 = scale1*masked_x1

#masked_x2 = model_masked2.predict(load_test_image(POS_IMAGE_SET_TEST, "2008_000512.jpg"))
#masked_x2 = scale2*masked_x2



# fiorellini
'''
print("[2] Computing raw model feature maps...")
print_feature_maps(raw_x, title="Conv1", n_imgs=4, cmap="rainbow")

print("[2] Computing masked model feature maps...")
print_feature_maps(masked_x, title="Masked(1)", n_imgs=4, cmap="rainbow")

print("[2] Computing masked final model feature maps...")
print_feature_maps(masked_x_final, title="Masked(2)", n_imgs=4, cmap="rainbow")

print("[2] Computing model comparison...")
print_comparison(raw_x, masked_x, n_imgs=2, cmap="rainbow")
'''
print("[2] Computing model comparison...")
print_comparison(raw_x1, masked_x1, n_imgs=2, cmap="rainbow")
'''
for i in range(5):
    print_comparison_step([raw_x1, masked_x1, raw_x2, masked_x2], n_imgs=4, cmap="rainbow", i=i)
print_comparison_step([raw_x1, masked_x1, raw_x2, masked_x2], n_imgs=4, cmap="rainbow", i=29)
'''
print("[2] Computing heatmaps...")
raw_heatmap1 = compute_heatmap(x=raw_x1, mode="avg")
masked_heatmap1 = compute_heatmap(x=masked_x1, mode="avg")

#raw_heatmap2 = compute_heatmap(x=raw_x2, mode="avg")
#masked_heatmap2 = compute_heatmap(x=masked_x2, mode="avg")

print_heatmap([20*raw_heatmap1, 40*masked_heatmap1], cmap="rainbow")
# print_heatmap([raw_heatmap1, 2*masked_heatmap1, raw_heatmap2/2, masked_heatmap2/2], cmap="rainbow")



# TODO:
#   - reshape della feature map alla stessa size dell'immagine di input (quadrata) e sovrapposizione per vedere che "zona" prende
#   - aggiungere altri tipi di maschera:
#       - L1 norm
#       - L2 norm   (lenta)
#       - ...
