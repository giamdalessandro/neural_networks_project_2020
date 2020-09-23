import os
import json
import tensorflow as tf
import numpy as np

from shutil import copy
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET = "./dataset/" 

TRAIN_VAL_PATH = DATASET + "train_val/"
ILSRVC_2013 = DATASET + "raw_data/detanimalpart/"
CUB_200 = DATASET + "raw_data/CUB_200_2011/"
PASCAL_VOC = DATASET + "raw_data/PascalVOC_2010_part/VOCdevkit/VOC2010/"

NUM_EPOCHS = 50
EPOCH_STEPS = 25
BATCH_SIZE = 16

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


def load_dataset(dataset="binary", batch_size=BATCH_SIZE, aug=False):
    """
    Loading training dataset via ImageDataGenerator
        - dataset:      one of 'binary', 'multi'
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
    if dataset == 'binary':
        data_path = TRAIN_VAL_PATH
    elif dataset == 'multi':
        # TODO
        raise NotImplementedError
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


def prepare_bin_dataset(dest_path=TRAIN_VAL_PATH):
    '''
        Copies PASCAL_VOC and CUB_200 bird images to 'train_val/bird/' directory,
        and PASCAL_VOC non-bird images to 'train_val/not_bird/' directory
    '''

    # PASCAL_VOC images, spillting birds from the others
    bird_imgs = []
    bird_ids = os.path.join(PASCAL_VOC, "ImageSets", "Main", "bird_trainval.txt")
    with open(bird_ids, "r") as f:
        for row in f.readlines():
            row_list = row.strip().split(' ')
            if len(row_list) == 3 and row_list[2] == "1":
                bird_imgs.append("{}.jpg".format(row_list[0]))

        f.close()
    print("... loaded {} bird images".format(len(bird_imgs)))

    b = not_b = 0
    voc_imgs = os.path.join(PASCAL_VOC, "JPEGImages")
    for img in os.listdir(voc_imgs):
        if img in bird_imgs:
            b += 1
            copy(src=os.path.join(voc_imgs, img), dst=os.path.join(dest_path, "bird"))
        
        else:
            not_b += 1
            copy(src=os.path.join(voc_imgs, img), dst=os.path.join(dest_path, "not_bird"))

    print("... copied {} non-bird images ...".format(not_b))
    print("... copied {} bird images ...DONE.".format(b))

    # CUB_200, just a symbolic link to the images directory
    os.system("cd dataset/train_val/bird/")
    os.system("ln -s ../../raw_data/CUB_200_2011/images/ cub_200_images")
    return None
