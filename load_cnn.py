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