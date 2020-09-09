from tensorflow.config import experimental
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from load_cnn import load_keras, load_dataeset

# GPU check
print("Num GPUs Available: ", len(experimental.list_physical_devices('GPU')))

''' TODO
    - Disentangled CNN
       - aggiungere loss filtri
    - Build decision trees

    1. loading pre-trained net from keras.Applications model, 
    'cause VGG16_vd .mat file is not working...
'''
model = Sequential(name="Interpretable_vgg16")

net = load_keras()
for layer in net.layers[:-1]:  # just exclude last layer from copying
    model.add(layer)
#print(model.summary())


''' 
    2. make top conv-layer(s) interpretable --> add masks to ouput filter
'''
top_conv = model.layers[-1]
out_tensor = top_conv.output
print(type(out_tensor), out_tensor)
print(out_tensor[0,0])
#block_mask_1 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", dilation_rate=(1,1),
#                    activation="relu", name="block_mask1")  # da completare con magia nera ?!?!
#block_mask_1.add_weight(shape=(3,3,512,512), initializer="random_normal", trainable=True)

#block_mask_1.add_loss()        # to build the interpretable tree?
#model.add(block_mask_1)         # add block_mask to model
model.add(MaxPool2D(name="block_pool", pool_size=(2,2),strides=(2,2)))         # add max pool layer

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
