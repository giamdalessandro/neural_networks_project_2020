import tensorflow as tf
from tensorflow.config import experimental
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from load_cnn import *

# GPU check
print("Num GPUs Available: ", len(experimental.list_physical_devices('GPU')))

'''
1. loading pre-trained net from keras.Applications model, because VGG16_vd .mat file is not working...
'''

model = load_keras()
print(model.summary())    # problema: toglie lo stato di input finale

'''
2. add loss for each of the 512 filters
'''

# skippata per ora

''' 
3. add masks to ouput filter
'''

# for all feature map in top conv layer:
n = 14
depth = 512
# out_tensor = tf.random.uniform(shape=(n,n,512), seed=42)
out_tensor = model.layers[-1].output[0]

for z in range(depth):
    feature_map = tf.slice(out_tensor, [0,0,z], [n,n,1])       # find max indices in the feature map
    #print("feature map:", feature_map)
    array_inutile = tf.reshape(feature_map, shape=[-1,1])
    #print(array_inutile)
    mu = my_argmax(array_inutile, n)
    #print("mu", mu)
    mask = compute_mask(mu, n)                             # compute mask centered in those indeces
    #print("mask", mask)
    output = tf.math.multiply(feature_map,mask)         # apply corresponding mask
    #print("masked output", output)
    
    # sostituire masked output all'output originale



'''
4. add final pooling
'''

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
