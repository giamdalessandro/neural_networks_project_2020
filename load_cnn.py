import json
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_keras():
    print("\n[1] Loading vgg16 from keras...")
    pretrained = VGG16(
        include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
        pooling=None, classes=1000, classifier_activation='softmax'
    )
    model = Sequential(name="Interpretable_vgg16")
    for layer in pretrained.layers[:-1]:  # just exclude last layer from copying
        model.add(layer)
    return model



def my_argmax(t, z):
    max_value = i_max = j_max = 0
    for i in range(14):
        for j in range(14):
            if t[i,j,z] > max_value:
                i_max = i
                j_max = j
                max_value = t[i,j,z]
    print("max value at depth %d found at %d%d : %d".format(z, i_max, j_max, max_value))
    return i_max, j_max



def compute_mask(mu):
    i_max = mu[0]
    j_max = mu[1]
    n    = 14
    tau  = 1                     # da verificare
    beta = 1                     # da verificare
    #mat = np.zeros(shape=(n,n,0))
    mat = tf.zeros(shape=(n,n,0))
    for i in range(n):
        for j in range(n):
            mat[i,j,0] = tau * max(-1, 1-beta*(abs(i-i_max)+abs(j-j_max))/n)
    print(mat)
    return mat

    











def load_very_deep():
    # c'è il casino da fare per tirare fuori i pesi
    # si può fare forse se si converte il .mat nel formato più nuovo (da matlab), ma per ora
    # non penso che serva
    from scipy.io import loadmat
    print("\n[1] Loading vgg16-verydeep from mat file...")
    net = loadmat("./dataset/imagenet-vgg-verydeep-16.mat") # load .mat file as a dict
    print("[1] {}".format(net.keys()))
    print("[1] loaded layers: {}".format(net["layers"].shape[1]))


    print("\n[1] Loading vgg16 from keras...")

    VGG16 = keras.applications.VGG16(
        include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
        pooling=None, classes=1000, classifier_activation='softmax')

    model = keras.Sequential(VGG16.layers[:-1])
    print(model.summary())

    print("[1] Model loaded.")
    #net_list = net["layers"][0].tolist()

    #import tables
    #file = tables.open_file("./dataset/imagenet-vgg-verydeep-16.mat", mode="r+")
    #print(len(file.root.layers[:]))
    
    # zavve-subgull & co. #
    
    for ar in net_list:
        layer = ar.tolist()
        print("zavve, {}, len: {}".format(type(ar),len(layer)))
        for l in layer:
            print("\tcol, {}, len: {}".format(type(l),len(l)))
            for i in l:
                print("\t\t", i)
                print("\t\tsubgull, {}, len: {}".format(type(i),len(i)))
                break
            break
        break

DATA_PATH = "./dataset/ILSVRC_2013_DET_part/"
BATCH_SIZE = 8

def load_dataeset():
    datagen = ImageDataGenerator(
        rescale=1./255,             # data agumentation 
        shear_range=0.2,
        zoom_range=0.2,       
        horizontal_flip=True,
        validation_split=0.2        # train and val
    )
    train_generator = datagen.flow_from_directory(
        directory=DATA_PATH,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        subset="training"
    )
    validation_generator = datagen.flow_from_directory(
        directory=DATA_PATH,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    return train_generator, validation_generator
