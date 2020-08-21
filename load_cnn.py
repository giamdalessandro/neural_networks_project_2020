import json
import tensorflow as tf
from tensorflow import keras

def load_cnn():
    print("\n[1] Loading vgg16 from keras...")
    VGG16 = keras.applications.VGG16(
        include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
        pooling=None, classes=1000, classifier_activation='softmax'
    )
    model = keras.Sequential(VGG16.layers[:-1])
    print(model.summary())

    print("\n[1] Loading vgg16-verydeep from mat file...")

    from scipy.io import loadmat
    net = loadmat("./dataset/imagenet-vgg-verydeep-16.mat") # load .mat file as a dict
    print("[1] {}".format(net.keys()))
    print("[1] loaded layers: {}".format(net["layers"].shape[1]))

    print("[1] Model loaded.")

    return net


def load_weights():
    from scipy.io import loadmat
    net = loadmat("./dataset/imagenet-vgg-verydeep-16.mat") # load .mat file as a dict
    print("[1] {}".format(net.keys()))
    
    net_list = net["layers"][0].tolist()
    #print(len(net_list))
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
    

load_weights()