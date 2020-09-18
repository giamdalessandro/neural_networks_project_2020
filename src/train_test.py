import tensorflow as tf
import matplotlib.image as mpimg
from datetime import datetime as dt

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from matplotlib import pyplot as plt

from maskLayer import *

from utils.visuallib import *
from utils.load_utils import load_keras, load_dataset

TRAIN       = True
MASK_LAYER  = True
NUM_EPOCHS  = 10
EPOCH_STEPS = 50
BATCH_SIZE  = 32
#FILTER_LOSS =    False

# GPU check
# print(tf.config.list_physical_devices('GPU'))

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
model.add(MaxPool2D(name="max_pool", pool_size=(2,2), strides=(2,2), data_format="channels_last"))
#print(model.summary())                         
model.trainable = False                         # we only train the top fully connected layers 


''' 
5. usare gli stessi FC di VGG16 inizializzati random 
'''
model.add(Flatten())
#model.add(Dropout(rate=0.8))
model.add(Dense(units=4096,activation="relu"))
#model.add(Dropout(rate=0.8))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=31, activation="softmax"))

print(model.summary())


'''
6. Compiling and training the model
'''
model.compile(
    optimizer= Adam(learning_rate=0.001),
    loss= categorical_crossentropy,
    metrics=["accuracy"]
)
print("[START TIME]: ",dt.now())

if TRAIN:
    train_generator, validation_generator = load_dataset(dataset='imagenet')
    model.fit(
        train_generator,
        steps_per_epoch=EPOCH_STEPS,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=100
    )
print("[END TIME]: ", dt.now())

model.save("model_multi_" + str(NUM_EPOCHS) + "_epochs_" +
            str(dt.now().day)    + "_" +
            str(dt.now().month)  + "_" +
            str(dt.now().year)   + "_" +
            str(dt.now().hour)   + "_" +    # serve?
            str(dt.now().minute) + ".h5")   # serve?

