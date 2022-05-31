# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 16:49:57 2022

@author: frede
"""

import datetime

from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import keras_tuner as kt


import os
os.chdir(r'C:\Users\frede\OneDrive\Desktop\Speciale\Programming')

from Helper_Functions.Custom_functions import training_functions, evaluation_functions

#%%%

TRAIN_PATH = 'SIIM-ISIC/train'
VAL_PATH = 'SIIM-ISIC/validation'
TEST_PATH = 'SIIM-ISIC/test'
SMALL_SIZE = (128,192)
FULL_SIZE = (512,768)


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 180)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_PATH,
                                                    target_size = SMALL_SIZE,
                                                    color_mode = 'rgb',
                                                    batch_size = 32,
                                                    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(VAL_PATH,
                                                        target_size = SMALL_SIZE,
                                                        color_mode = 'rgb',
                                                        batch_size = 16,
                                                        class_mode = 'categorical')

TRAIN_N = 36670
VAL_N = 6471

#%% Setting up the model

def model_builder(hp):
    
    # Setting up the pre-trained part of the network
    
    base = EfficientNetB3(include_top = False,
                 input_shape = (192,158,3))
    base.trainable = False
    
    # Defining the kernel reguralizer parameters - they are used twice
    
    model = Sequential()
    model.add(base)
    model.add(Flatten())
    if hp.Choice('Dropout', values=[True, False]) == True:
        model.add(Dropout(0.5,))
    model.add(Dense(hp.Int('hidden_1_size', 64, 512, step = 64),
                    activation = 'relu',
                    kernel_regularizer = l2(hp.Float('hidden_1_reg', 0.0, 0.1)),
                    name = 'Dense1'))
    if hp.Choice('Batch_norm', values=[True, False]) == True:
        model.add(BatchNormalization())
    model.add(Dense(hp.Int('hidden_2_size', 32, 256, step = 32),
                    activation = 'relu',
                    kernel_regularizer = l2(hp.Float('hidden_2_reg', 0.0, 0.1)),
                    name = 'Dense2'))
    model.add(Dense(8, activation = 'softmax'))
    
    model.compile(optimizer = Adam(learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling = 'log')),
                  loss = 'categorical_crossentropy',
                  metrics = AUC(multi_label = True))
    
    return model

#%%

# Running the search algorithm

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [EarlyStopping(monitor='val_loss', patience = 5),
             ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3),
             TensorBoard(logdir, histogram_freq = 1)]

tuner = kt.Hyperband(model_builder,
                     objective = 'val_loss',
                     max_epochs = 15,
                     factor = 3,
                     directory = 'Hyperband',
                     project_name = 'EfficientNetV3')

tuner.search(train_generator,
             steps_per_epoch = 10,
             validation_data = validation_generator,
             validation_steps = 5,
             callbacks = callbacks)

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('Tuning/Tuned_model')

#%% Unfreezing and retraining the best model

if not best_model in locals():
    best_model = evaluation_functions.load_model('Hyperband/Best_model')
    
best_model.trainable = True

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [EarlyStopping(monitor='val_loss', patience = 5),
             ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3),
             TensorBoard(logdir, histogram_freq = 1)]

best_model.fit(train_generator,
               steps_per_epoch = 10,
               validation_data = validation_generator,
               validation_steps = 5,
               callbacks = callbacks)

best_model.save('Tuning/Retrained_best_model')

#%% Summary of the best model

model = evaluation_functions.load_model('Hyperband/Retrained_best_model')

print(model.summary())

#%% Upscaling picture size and model to see number of parameters

TRAIN_PATH = 'SIIM-ISIC/train'
VAL_PATH = 'SIIM-ISIC/validation'
TEST_PATH = 'SIIM-ISIC/test'
SMALL_SIZE = (192,158)
FULL_SIZE = (768,512)


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 180)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(TRAIN_PATH,
                                                    target_size = FULL_SIZE,
                                                    color_mode = 'rgb',
                                                    batch_size = 32,
                                                    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(VAL_PATH,
                                                        target_size = FULL_SIZE,
                                                        color_mode = 'rgb',
                                                        batch_size = 16,
                                                        class_mode = 'categorical')

TRAIN_N = 36670
VAL_N = 6471

def full_model():
    
    base = EfficientNetB3(include_top = False,
                 input_shape = (512,768,3))
    base.trainable = False
    
    # Defining the kernel reguralizer parameters - they are used twice
    
    model = Sequential()
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(8, activation = 'softmax'))
    
    model.compile(optimizer = Adam(),
                  loss = 'categorical_crossentropy',
                  metrics = AUC(multi_label = True))
    
    return model

big_model = full_model()

print(big_model.summary())











