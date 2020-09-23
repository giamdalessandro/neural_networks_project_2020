import tensorflow as tf
import matplotlib.image as mpimg

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from matplotlib import pyplot as plt

from maskLayer import *

from utils.visuallib import *
from utils.dataset_utils import load_keras


model_raw1 = load_keras()

model_masked1 = load_keras()
model_masked1.add(MaskLayer())

model_raw2 = load_keras()
model_raw2.add(MaskLayer())
model_raw2.add(Conv2D(512, [3, 3], padding="same",
                  activation='relu', name="our_conv"))

model_masked2 = load_keras()
model_masked2.add(MaskLayer())
model_masked2.add(Conv2D(512, [3, 3], padding="same",
                  activation='relu', name="our_conv"))
model_masked2.add(MaskLayer())


raw_x1 = model_raw1.predict(load_def())
raw_x2 = model_raw2.predict(load_def())
raw_x2 = 1000 * raw_x2

masked_x1 = model_masked1.predict(load_def())  # we use their parameters but scale the masked x before plotting for visual reason
masked_x1 = scale1*masked_x1

masked_x2 = model_masked2.predict(load_def())
masked_x2 = scale2*masked_x2



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

print("[2] Computing model comparison...")
print_comparison(masked_x, masked_x_final, n_imgs=2, cmap="rainbow")
'''
for i in range(5):
    print_comparison_step([raw_x1, masked_x1, raw_x2, masked_x2], n_imgs=4, cmap="rainbow", i=i)
print_comparison_step([raw_x1, masked_x1, raw_x2, masked_x2], n_imgs=4, cmap="rainbow", i=29)

print("[2] Computing heatmaps...")
raw_heatmap1 = compute_heatmap(x=raw_x1, mode="avg")
masked_heatmap1 = compute_heatmap(x=masked_x1, mode="avg")

raw_heatmap2 = compute_heatmap(x=raw_x2, mode="avg")
masked_heatmap2 = compute_heatmap(x=masked_x2, mode="avg")

print_heatmap([raw_heatmap1, 2*masked_heatmap1, raw_heatmap2/2, masked_heatmap2/2], cmap="rainbow")



# TODO:
#   - reshape della feature map alla stessa size dell'immagine di input (quadrata) e sovrapposizione per vedere che "zona" prende
#   - aggiungere altri tipi di maschera:
#       - L1 norm
#       - L2 norm   (lenta)
#       - ...


'''
4. add final pooling


model.add(MaxPool2D(name="block_pool", pool_size=(2,2),strides=(2,2)))         # add max pool layer
print(model.summary())    # problema: toglie lo stato di input finale

'''











'''
    3. aggiungere filter loss al top_conv_layer
'''
#block_mask_2 = 
#model.add(block_mask_2)         # add block_mask_2 to model


''' 
    4. aggiungere strato flatten?
'''
##model.add(Flatten())

''' 
    5. usare gli stessi FC inizializzati random 
'''
##model.add(Dropout(rate=0.8))
#model.add(Dense(units=4096,activation="relu"))
##model.add(Dropout(rate=0.8))
#model.add(Dense(units=4096,activation="relu"))
#model.add(Dense(units=2, activation="softmax"))
#
#print(model.summary())
#
#model.compile(
#    optimizer= Adam(learning_rate=0.01),
#    loss= categorical_crossentropy,
#    metrics=["accuracy"]
#)
#train_generator, validation_generator = load_dataeset()
#model.fit(
#    train_generator,
#    steps_per_epoch=50,
#    epochs=10,
#    validation_data=validation_generator,
#    validation_steps=100
#)
