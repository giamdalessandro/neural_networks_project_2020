import json
from tensorflow import keras

def load_keras():
    print("\n[1] Loading vgg16 from keras...")
    model = keras.applications.VGG16(
        include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
        pooling=None, classes=1000, classifier_activation='softmax'
    )
    #model = keras.Sequential(VGG16.layers[:-1])
    print(model.summary())

    return model


def load_very_deep():
    from scipy.io import loadmat
    print("\n[1] Loading vgg16-verydeep from mat file...")
    net = loadmat("./dataset/imagenet-vgg-verydeep-16.mat") # load .mat file as a dict
    print("[1] {}".format(net.keys()))
    print("[1] loaded layers: {}".format(net["layers"].shape[1]))

    print("[1] Model loaded.")
    #net_list = net["layers"][0].tolist()

    #import tables
    #file = tables.open_file("./dataset/imagenet-vgg-verydeep-16.mat", mode="r+")
    #print(len(file.root.layers[:]))

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
    
# .test
#load_keras()
#load_very_deep()