from tensorflow.config import experimental
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from load_cnn import load_keras

# GPU check
print("Num GPUs Available: ", len(experimental.list_physical_devices('GPU')))

# TODO
#   - Disentangled CNN
#       - aggiungere loss filtri
#   - Build decision trees

#   1. loading pre-trained net from keras.Applications model, 
#   'cause VGG16_vd .mat file is not working...

model = Sequential(name="Interpretable_vgg16")

net = load_keras()
for layer in net.layers[:-1]:  # just exclude last layer from copying
    model.add(layer)
#print(model.summary())


#   2. modificare filtri nel top conv-layer --> aggiungere maschere
#   3. aggiungere un nuovo conv-layer con M=512 filtri --> ogni filtro è un tensore 3x3xM
#   - trying to add block_mask black magic  
block_mask_1 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", dilation_rate=(1,1),
                    activation="relu", name="block_mask1")  # da completare con magia nera ?!?!
block_mask_1.add_weight(shape=(3,3,512,512), initializer="random_normal", trainable=True)
#block_mask_1.add_loss()                                    # to build the interpretable tree?

model.add(block_mask_1)                                     # add block_mask to model
model.add(MaxPool2D(name="block_pool", pool_size=(2,2),strides=(2,2)))         # add max pool layer


#   4. aggiungere maschere per i filtri del nuovo conv-layer
block_mask_2 = Conv2D(filters=4096, kernel_size=(3,3), strides=(1,1), padding="valid", dilation_rate=(1,1),
                    activation="relu", name="block_mask2")  # conv or fully-connected ?!?!
block_mask_2.add_weight(shape=(7,7,512,4096), initializer="random_normal", trainable=True)
#block_mask_2.add_loss()                                    # to build the interpretable tree?

model.add(block_mask_2)                                     # add block_mask_2 to model
print(model.summary())


#   5. usare gli stessi FC inizializzati random