# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:49:53 2022

@author: frede
"""


import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.metrics import AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import os
os.chdir(os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

from Helper_Functions.Custom_functions import training_functions, evaluation_functions


#%% Setting up data import using ImageDataGenerator

TRAIN_PATH = 'SIIM-ISIC_binary/train'
VAL_PATH = 'SIIM-ISIC_binary/validation'
TEST_PATH = 'SIIM-ISIC/test'
FULL_SIZE = (512,768)
TEST_SIZE = (256,384)


batch = 64

train_datagen = ImageDataGenerator(#rescale = 1./255,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 180)

test_datagen = ImageDataGenerator()#rescale = 1./255

train_generator = train_datagen.flow_from_directory(TRAIN_PATH,
                                                    target_size = TEST_SIZE,
                                                    color_mode = 'rgb',
                                                    batch_size = batch,
                                                    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(VAL_PATH,
                                                        target_size = TEST_SIZE,
                                                        color_mode = 'rgb',
                                                        batch_size = batch,
                                                        class_mode = 'binary')

test_generator = test_datagen.flow_from_directory(TEST_PATH,
                                                  target_size = TEST_SIZE,
                                                  color_mode = 'rgb',
                                                  batch_size = 1,
                                                  class_mode = None,
                                                  shuffle = False)

# Number of steps per epoch

train_steps = int(train_generator.samples / batch)
val_steps = int(validation_generator.samples / batch)
test_steps = int(test_generator.samples / batch)

# Instance distribution in training and validation data

os.makedirs('SIIM-ISIC_binary/Descriptives', exist_ok = True)

training_dist = training_functions.category_distribution(TRAIN_PATH)
training_dist.to_csv('SIIM-ISIC_binary/Descriptives/Training_dist.csv')
validation_dist = training_functions.category_distribution(VAL_PATH)
validation_dist.to_csv('SIIM-ISIC_binary/Descriptives/Validation_dist.csv')

mel_num = 0

#%% Also preparing evaluation dataset

truth_path = 'Evaluation data/Diagnoses.xls'
guess_path = 'Evaluation data/Full database anonymous.xlsx'
data, _, _, _ = evaluation_functions.csv_import(truth_path, guess_path)

im_path = 'Evaluation data/Skin cancer images'
X_eval, y_eval, idx = evaluation_functions.import_X_and_y(im_path, TEST_SIZE, data)

#%% Defining model

def build_model(drop):
    
    base = EfficientNetB3(include_top = False,
                          weights = 'imagenet',
                          input_shape=(256,384,3),
                          pooling = 'avg')
    
    model = Sequential()
    model.add(base)
    if drop != 0.0:
        model.add(Dropout(drop))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(optimizer = Adam(),
                  loss = 'binary_crossentropy',
                  metrics = AUC(name='AUC'))
    
    return model


#%% Setting class weights, learning rate schedule, callbacks and grid.

total_num = training_dist['Count'].sum()
malignant_num = training_dist[training_dist['Category'] == 'malignant']['Count'].sum()
benign_num = training_dist[training_dist['Category'] == 'benign']['Count'].sum()

# Setting class weights

weight_for_0 = 1/benign_num*total_num
weight_for_1 = 1/malignant_num*total_num

class_weight = {0: weight_for_0,
                1: weight_for_1}

# Defining a learning rate schedule from a custom function

lr_schedule = training_functions.get_lr_callback(batch_size = batch)

dropout_grid = [0,0.2,0.4,0.6]


#%% Running the small grid

for drop in dropout_grid:
    
    # Setting up output string and callbacks
    
    output_str = f'Binary-shallow/dropout_{drop}'
    os.makedirs(output_str, exist_ok = True)
    csv_str = output_str+'/eval_files'
    os.makedirs(csv_str, exist_ok = True)
    model_str = output_str+'/model'
    os.makedirs(model_str, exist_ok = True)
    
    log_dir = "logs/fit/" + output_str
    
    callback_list = [lr_schedule,
                     ModelCheckpoint(model_str,
                                     save_best_only=True,
                                     save_weights_only=False),
                     TensorBoard(log_dir=log_dir,
                                 histogram_freq =1),
                     EarlyStopping(patience = 7,
                                   restore_best_weights = True)]
    
    # Initializing and training a model
    
    model = build_model(drop)
    model.fit(train_generator,
              steps_per_epoch = train_steps,
              validation_data = validation_generator,
              validation_steps = val_steps,
              epochs = 50,
              class_weight = class_weight,
              callbacks = callback_list)
    
    # Evaluating on the SIIM-ISIC data to see if training has occured
    
    train_df, _ = evaluation_functions.prediction_dataframes(model, True, train_generator, train_generator)
    train_df.to_csv(csv_str+'/train_df.csv', index = False)
    
    test_df, submit_df = evaluation_functions.prediction_dataframes(model, True, train_generator, test_generator)
    test_df.to_csv(csv_str+'/test_df.csv', index = False)
    submit_df.to_csv(csv_str+'/submit.csv', index = False)
    
    # Evaluating on the Zadaulek data
    
    eval_df, eval_target_df = evaluation_functions.Zadaulek_eval(model, X_eval, idx, mel_num)
    eval_df.to_csv(csv_str+'/eval_full.csv')
    eval_target_df.to_csv(csv_str+'/eval_target.csv')
        



