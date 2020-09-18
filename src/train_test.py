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

NUM_EPOCHS  = 100
EPOCH_STEPS = 50
BATCH_SIZE  = 32


'''
1. loading pre-trained net from keras.Applications model, because VGG16_vd .mat file is not working...
'''


fc = [
    MaxPool2D(name="max_pool", pool_size=(2, 2),strides=(2, 2), data_format="channels_last"),
    Flatten(),
    Dense(units=4096, activation="relu"),
    Dropout(rate=0.8),
    Dense(units=4096, activation="relu"),
    Dropout(rate=0.8), 
    Dense(units=31, activation="softmax")]

### MODEL RAW ###
model_raw = load_keras()
model_raw.trainable = False
for i in fc:
    model_raw.add(i)
model_raw.summary()


### MODEL MASKED 1 ###
model_masked1 = load_keras()
model_masked1.add(MaskLayer())
model_masked1.trainable = False
for i in fc:
    model_masked1.add(i)
model_masked1.summary()

'''

### MODEL MASKED 2 ###
model_masked2 = load_keras()
model_masked2.add(MaskLayer())
model_masked2.trainable = False
model_masked2.add(Conv2D(512, [3, 3], padding="same", activation='relu', name="our_conv"))
model_masked2.add(MaskLayer())
for i in fc:
    model_masked2.add(i)
model_masked2.summary()

'''

'''
6. Compiling and training the model
'''
model_raw.compile(
    optimizer= Adam(learning_rate=0.001),
    loss= categorical_crossentropy,
    metrics=["accuracy"]
)

model_masked1.compile(
    optimizer= Adam(learning_rate=0.001),
    loss= categorical_crossentropy,
    metrics=["accuracy"]
)
'''
model_masked2.compile(
    optimizer= Adam(learning_rate=0.001),
    loss= categorical_crossentropy,
    metrics=["accuracy"]
)
start = dt.now()
'''
print("[START TIME]: ",dt.now())
train_generator, validation_generator = load_dataset(dataset='imagenet')
model_raw.fit(
    train_generator,
    steps_per_epoch=EPOCH_STEPS,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=100
)
print("[END TIME]: ", dt.now())


print("[START TIME]: ", dt.now())
#train_generator, validation_generator = load_dataset(dataset='imagenet')
model_masked2.fit(
    train_generator,
    steps_per_epoch=EPOCH_STEPS,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=100
)
print("[END TIME]: ", dt.now())

'''
print("[START TIME]: ", dt.now())
#train_generator, validation_generator = load_dataset(dataset='imagenet')
model_masked2.fit(
    train_generator,
    steps_per_epoch=EPOCH_STEPS,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=100
)
print("[END TIME]: ", dt.now())
'''
end = dt.now()

print ("ELAPSED TIME: ", end - start)
print(start)
print(end)

model_raw.save("raw_multi_" + str(NUM_EPOCHS) + "_epochs_" +
            str(dt.now().day)    + "_" +
            str(dt.now().month)  + "_" +
            str(dt.now().year)   + "_" +
            str(dt.now().hour)   + "_" +    # serve?
            str(dt.now().minute) + ".h5")   # serve?

model_masked1.save("msk1_multi_" + str(NUM_EPOCHS) + "_epochs_" +
            str(dt.now().day)    + "_" +
            str(dt.now().month)  + "_" +
            str(dt.now().year)   + "_" +
            str(dt.now().hour)   + "_" +    # serve?
            str(dt.now().minute) + ".h5")   # serve?
'''
model_masked2.save("msk2_multi_" + str(NUM_EPOCHS) + "_epochs_" +
               str(dt.now().day) + "_" +
               str(dt.now().month) + "_" +
               str(dt.now().year) + "_" +
               str(dt.now().hour) + "_" +    # serve?
               str(dt.now().minute) + ".h5")   # serve?
'''