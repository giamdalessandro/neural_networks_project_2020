import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from load_cnn import load_keras

# GPU check
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# TODO
#   - Disentangled CNN
#       - aggiungere loss filtri
#   - Build decision trees

#   1. loading pre-trained net from keras.Applications model, 
#   'cause VGG16_vd .mat file is not working...

net = load_keras()
model = keras.Sequential()
for layer in net.layers[:-1]:  # just exclude last layer from copying
    model.add(layer)
#print(model.summary())


#   2. modificare filtri nel top conv-layer --> aggiungere maschere
#   3. aggiungere un nuovo conv-layer con M=512 filtri --> ogni filtro è un tensore 3x3xM
#   - trying to add block_mask black magic  
block_mask_1 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", dilation_rate=(1,1),
                    activation="relu", name="block_mask1")  # da completare con magia nera ?!?!

model.add(block_mask_1)                                     # add block_mask to model
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))         # add max pool layer

block_mask_2 = Conv2D(filters=4096, kernel_size=(3,3), strides=(1,1), padding="valid", dilation_rate=(1,1),
                    activation="relu", name="block_mask2")  # conv or fully-connected ?!?!

model.add(block_mask_2)                                     # add block_mask_2 to model
print(model.summary())


#   4. aggiungere maschere per i filtri del nuovo conv-layer
#   5. usare gli stessi FC inizializzati random