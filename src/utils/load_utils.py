import json
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ILSRVC_2013 = "./dataset/detanimalpart/"
CUB_200 = "./dataset/CUB_200_2011/"
PASCAL_VOC = "./dataset/PascalVOC_2010_part/VOCdevkit/"

# ne fa 100 all'ora
NUM_EPOCHS = 100
EPOCH_STEPS = 50
BATCH_SIZE = 32

def load_keras(name="our_interpretable_cnn"):
    print("\n[1] Loading vgg16 from keras...")
    pretrained = VGG16(
        include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3),
        pooling=None, classes=1000, classifier_activation='softmax'
    )
    model = Sequential(name=name)
    for layer in pretrained.layers[:-1]:  # just exclude last layer from copying
        model.add(layer)
    print("[1]                        Loaded.")
    return model


def load_dataset(dataset='imagenet', batch_size=BATCH_SIZE, aug=False):
    """
    Loading training dataset via ImageDataGenerator
        - dataset:      one of 'cub200', 'imagenet', 'voc2010'
        - batch_size:   data batch size for training
    """
    if aug:
        datagen = ImageDataGenerator(
            rescale=1./255,             # data aug 
            shear_range=0.2,
            zoom_range=0.2,       
            horizontal_flip=True,
            validation_split=0.2        # train and val
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1./255,             # train and val
            validation_split=0.2        
        )

    data_path = ''
    if dataset == 'imagenet':
        data_path = ILSRVC_2013
    elif dataset == 'cub200':
        data_path = CUB_200
    elif dataset == 'voc2010':
        data_path = PASCAL_VOC
    print("\nUsing " + dataset + " dataset for training...")
    

    train_generator = datagen.flow_from_directory(
        directory=data_path,
        target_size=(224,224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset="training"
    )
    validation_generator = datagen.flow_from_directory(
        directory=data_path,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    return train_generator, validation_generator


