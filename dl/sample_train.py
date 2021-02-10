import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Input, MaxPooling2D, Dropout, Flatten
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import  keras.backend as K
from keras.applications.mobilenet_v2 import MobileNetV2

import random
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

weight_decay = 1e-4


def make_mbnet_model():
    K.clear_session()
    mbv2_model = MobileNetV2(input_shape=(224, 224, 3),
                             include_top=False,
                             weights='imagenet')
    
    top_model = Sequential()
    top_model.add(Flatten(input_shape=mbv2_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Model(
        inputs=mbv2_model.input,
        outputs=top_model(mbv2_model.output)
    )
    model.compile(optimizer=SGD(lr=0.001,momentum=0.9),
          loss='binary_crossentropy',
          metrics=['accuracy'])
    mbv2_model.trainable = False
    return model



def train(dataset_dir, model_save_path):

    model = make_mbnet_model()
    model.summary()


    BATCH_SIZE = 16
    TRAIN_NUM = 30294
    train_datagen = ImageDataGenerator(
                                        rescale=1./255,
                                    rotation_range=180,
    #                             width_shift_range = 0.05,
    #                             height_shift_range = 0.05,
    #                             shear_range = 0.05,
                                horizontal_flip = True,
                                vertical_flip = True
                                )

    val_datagen = ImageDataGenerator(
        rescale=1./255
        )

    train_generator = train_datagen.flow_from_directory(
                dataset_dir + 'train/',
                target_size=(224,224),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                shuffle=True
            )

    val_generator = val_datagen.flow_from_directory(
                dataset_dir + 'val/',
                target_size=(224,224),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                shuffle=False
            )



    history = model.fit_generator(train_generator, shuffle=True, steps_per_epoch =TRAIN_NUM // BATCH_SIZE,
                              epochs = 20,
                    validation_data = val_generator, validation_steps = 1, callbacks=[
                                        ModelCheckpoint(filepath = model_save_path,
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='min',
                                  period=1),
                                   ReduceLROnPlateau(
                                                 monitor='val_loss',
                                                 patience=10,
                                                 verbose=1,
                                                 factor=0.9,
                                                 min_lr = 0.00001)])

