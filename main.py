import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from load_utils import load_keras
from oldMaskLayer import *
from visualize import *
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# GPU check
# print(tf.test.is_gpu_available())

'''
1. loading pre-trained net from keras.Applications model, because VGG16_vd .mat file is not working...
'''

model_raw = load_keras()
model_masked = load_keras()


'''
2. add loss for each of the 512 filters
'''

# skippata per ora

''' 
3. add masks to ouput filter
'''

visualize = False       # False = use their parameters (but scale the masked x before plotting)
                        # True = use dummy parameters (t=b=1), but no need to scale (better for seeing the differences)
scale = 100

print("\n[2] Adding one mask layer...")
model_masked.add(OldMaskLayer(visualize=visualize))
print("[2]                        Added.")

# ORSETTO LAVAROSSO VA IN CERCA DI FIORELLINI #

raw_x = model_raw.predict(load_def())
masked_x = model_masked.predict(load_def())

if visualize == False:
    masked_x = scale*masked_x


print("[2] Computing raw model feature maps...")
print_feature_maps(raw_x, masked=False, n_imgs=4, cmap="rainbow")

print("[2] Computing masked model feature maps...")
print_feature_maps(masked_x, masked=True, n_imgs=4, cmap="rainbow")

print("[2] Computing model comparison...")
print_comparison(raw_x, masked_x, n_imgs=2, cmap="rainbow")
for i in range(20):
    print_comparison_step(raw_x, masked_x, n_imgs=4, cmap="rainbow", i=i)


print("[2] Computing heatmaps...")
raw_heatmap = compute_heatmap(x=raw_x, mode="avg")
masked_heatmap = compute_heatmap(x=masked_x, mode="avg")
print_heatmap(raw_heatmap, masked_heatmap, scale="same", cmap="rainbow")



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
