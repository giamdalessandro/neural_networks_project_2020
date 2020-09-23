import tensorflow as tf
import matplotlib.image as mpimg
from datetime import datetime as dt

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

from matplotlib import pyplot as plt

from maskLayer import *

from utils.visuallib import *
from utils.dataset_utils import *

model_list = []

fc_no_dropout=[
    MaxPool2D(name="max_pool", pool_size=(2, 2),strides=(2, 2), data_format="channels_last"),
    Flatten(),
    Dense(units=4096, activation="relu"),
    #Dropout(rate=0.8),
    Dense(units=4096, activation="relu"),
    #Dropout(rate=0.8), 
    Dense(units=2, activation="softmax")]

fc= [
    MaxPool2D(name="max_pool", pool_size=(2, 2),strides=(2, 2), data_format="channels_last"),
    Flatten(),
    Dense(units=4096, activation="relu"),
    Dropout(rate=0.8),
    Dense(units=4096, activation="relu"),
    Dropout(rate=0.8),
    Dense(units=2, activation="softmax")]

''' MODEL RAW
model_raw = load_keras(name="raw")
model_raw.trainable = False
for i in fc:
    model_raw.add(i)
model_raw.summary()
model_list.append(model_raw)
'''


''' MODEL MASKED 1 NO DROPOUT '''
model_masked1_no_dropout = load_keras(name="masked1_no_dropout")
model_masked1_no_dropout.add(MaskLayer())
model_masked1_no_dropout.trainable = False
for i in fc_no_dropout:
    model_masked1_no_dropout.add(i)
model_masked1_no_dropout.summary()
model_list.append(model_masked1_no_dropout)

''' MODEL MASKED 1 WITH DROPOUT '''
model_masked1 = load_keras(name="masked1")
model_masked1.add(MaskLayer())
model_masked1.trainable = False
for i in fc:
    model_masked1.add(i)
model_masked1.summary()
model_list.append(model_masked1)

'''
###  MODEL MASKED 2 ###
model_masked2 = load_keras(name="masked2")
model_masked2.add(MaskLayer())
model_masked2.trainable = False
model_masked2.add(Conv2D(512, [3, 3], padding="same", activation='relu', name="our_conv"))
model_masked2.add(MaskLayer())
for i in fc:
    model_masked2.add(i)
model_masked2.summary()
model_list.append(model_masked2)

'''

for m in model_list:
    m.compile(
        optimizer= Adam(learning_rate=0.001),
        loss= categorical_crossentropy,
        metrics=["accuracy"]
    )

start = dt.now()
train_generator, validation_generator = load_dataset(dataset='binary')

for m in model_list:
    print("[START TIME]: ", dt.now())
    history = m.fit(
        train_generator,
        steps_per_epoch=EPOCH_STEPS,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=100)
    
    print("[END TIME]: ", dt.now())
    m.save( "./models/" + m.name + "_binary_"  + 
            str(NUM_EPOCHS) +   "_epochs_" +
            str(dt.now().day)    + "_" +
            str(dt.now().month)  + "_" +
            str(dt.now().year)   + "_" +
            str(dt.now().hour)   + "_" +    # serve?
            str(dt.now().minute) + ".h5", save_format="h5")   # serve?

    #pretty_little_flower(history)

end = dt.now()

print("ELAPSED TIME: ", end - start)
print(start)
print(end)
