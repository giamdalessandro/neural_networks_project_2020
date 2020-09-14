import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from load_cnn import load_keras
from maskLayer import *
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
print("\n[2] Adding one mask layer...")
model_masked.add(MaskLayer())
print("[2]                        << Added.")

# ORSETTO LAVAROSSO VA IN CERCA DI FIORELLINI #
#raw_heatmap = compute_heatmap(model=model_raw, img=loaddd(), mask=False)
#masked_heatmap = compute_heatmap(model=model_masked, img=loaddd(), mask=True)
#print_heatmap(raw_heatmap, masked_heatmap)
raw_x = model_raw.predict(loaddd())
masked_x = model_masked.predict(loaddd())
print("[2] Computing raw model feature maps...")
print_feature_maps(raw_x, masked=False, n_imgs=2)

print("[2] Computing masked model feature maps...")
print_feature_maps(masked_x, masked=True, n_imgs=2)

print("[2] Computing model comparison...")
print_comparison(raw_x, masked_x, n_imgs=2)

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
