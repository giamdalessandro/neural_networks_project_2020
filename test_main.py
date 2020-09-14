import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from load_utils import load_keras, load_dataeset
from maskLayer import *

MASK_LAYER = 1
FILTER_LOSS = 0
# GPU check
# print(tf.test.is_gpu_available())

'''
1. loading pre-trained net from keras.Applications model, because VGG16_vd .mat file is not working...
'''
model = load_keras()


'''
2. TODO add loss for each of the 512 filters
'''


''' 
3. add masks to ouput filter
'''
if MASK_LAYER:
    model.add(MaskLayer())


'''
4. add final pooling
'''
model.add(MaxPool2D(name="block_pool", pool_size=(2,2),strides=(2,2)))         # add max pool layer
#print(model.summary())                         # problema: toglie lo stato di input finale
model.trainable = False                         # we only train the top fully connected layers 

''' 
5. usare gli stessi FC di VGG16 inizializzati random 
'''
model.add(Flatten())
#model.add(Dropout(rate=0.8))
model.add(Dense(units=4096,activation="relu"))
#model.add(Dropout(rate=0.8))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2, activation="softmax"))

print(model.summary())


'''
6. Compiling and training the model
'''
model.compile(
    optimizer= Adam(learning_rate=0.01),
    loss= categorical_crossentropy,
    metrics=["accuracy"]
)
#train_generator, validation_generator = load_dataeset(dataset='cub200')
#model.fit(
#    train_generator,
#    steps_per_epoch=50,
#    epochs=10,
#    validation_data=validation_generator,
#    validation_steps=100
#)