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

model1 = load_keras()
#model1.add(Activation(tf.keras.activations.tanh))
#print(model.summary())    # problema: toglie lo stato di input finale
feature_map_no_mask = compute_heatmap(model=model1, img=loaddd(), mask=False)

'''
2. add loss for each of the 512 filters
'''

# skippata per ora

''' 
3. add masks to ouput filter
'''

model2 = load_keras()
model2.add(MaskLayer())
#model2.add(Activation(tf.keras.activations.relu))
feature_map_mask = compute_heatmap(model=model2, img=loaddd(), mask=True)

print_heatmap(feature_map_no_mask, feature_map_mask)



# vedi qui https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/multi_image.html#sphx-glr-gallery-images-contours-and-fields-multi-image-py

# printa vicino le feature maps mascherate e non per capire che cazzo succede




print_feature_maps(model1, masked=False)
print_feature_maps(model2, masked=True)

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
