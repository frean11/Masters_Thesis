# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:03:16 2022

@author: frede
"""


from numpy import expand_dims
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import glob

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

import os
os.chdir(os.path.dirname(__file__))
os.makedirs('Visualizations', exist_ok = True)

from Helper_Functions.Custom_functions import evaluation_functions


#%% Showing images from the Data generator

try:
    os.mkdir('SIIM-ISIC_binary/Augmentation')
except: 
    pass

image_name = 'ISIC_0000482.jpg'
img = load_img('SIIM-ISIC_binary/train/malignant/'+image_name)
data = img_to_array(img)
data = expand_dims(data, 0)

datagen = ImageDataGenerator(zoom_range = 0.2,
                             horizontal_flip = True,
                             rotation_range = 180,
                             shear_range = 0.2)
iterator = datagen.flow(data, batch_size=1)

for i in range(9):
    plt.subplot(330+1+i)
    batch = iterator.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
    plt.axis('off')
plt.suptitle('Augmented versions of image 0000482')
plt.savefig('Visualizations/Augmented_0000482.jpg')
plt.show()


#%% Importing Zadaulek and images with labels and indexes

truth_path = 'Evaluation data/Diagnoses.xls'
guess_path = 'Evaluation data/Full database anonymous.xlsx'
data, useful_participants, professional_grouping, experience_grouping = evaluation_functions.csv_import(truth_path, guess_path)

im_path = 'Evaluation data/Skin cancer images'
im_shape = (256,256)
X_eval, y_eval, idx = evaluation_functions.import_X_and_y(im_path, im_shape, data)
idx = [int(x) for x in idx]

#%% Getting descriptives for the participants in the Zadaulek study

participants = []

for i in data['OBSERVER ID'].unique():
    dat = data[data['OBSERVER ID'] == i]
    dat_len = len(dat)
    train_len = len(dat[dat['TRAINING'] == 'yes'])
    test_len = dat_len-train_len
    if test_len == 0:
        test_cat = '0'
    elif test_len > 0 and test_len <= 25:
        test_cat = '1-25'
    elif test_len > 25 and test_len <= 75:
        test_cat = '26-75'
    elif test_len > 75 and test_len <= 125:
        test_cat = '76-125'
    else:
        test_cat = '126-150'
    
    profession = dat['OBSERVER PROFESSION'].iloc[0]
    
    participants.append((profession, test_cat))

scoring = {}

participant_set = set(participants)
for j in participant_set:
    scoring[j] = participants.count(j)


# Calculating and plotting TPR and FPR for every participant along with average performance

x_values, x_avg, y_values, y_avg = evaluation_functions.human_fpr_tpr(data, useful_participants, idx)

print(f'The average human performance has fpr {x_avg} and tpr {y_avg}')

plt.scatter(x_values, y_values, c = 'grey', alpha = 0.5, label = 'Individual performances')
plt.scatter(x_avg, y_avg, c = 'blue', label = 'Human baseline')
plt.legend(loc = 'lower right')
plt.title('Performance of experienced dermatologists')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('Visualizations/Dermatologist_performance.png')
plt.show()

                
#%% Getting evaluation performances for each model in final setup

targets = glob.glob(r'Trained models/**/**/**/**/*eval_target.csv')

output = []

for i in tqdm(targets):
    file = pd.read_csv(i)
    predictions = [x for x in file['prediction']]
    df, auc = evaluation_functions.ROC_eval(predictions, y_eval)
    output.append([i, auc])

output_df = pd.DataFrame(output)


#%% Creating plot of training and validation AUC and loss for top performing model (Multiclass, Deep, Dropout = 0.4)

# CSV files have been downloaded from Tensorboard for the top performing model and saved into a folder named "Trained models"

files = glob.glob('Trained models/*.csv')

# Creating the X and Y values for the plots

X = {}
Y = {}

for i in files:
    csv = pd.read_csv(i)
    x = [x+1 for x in csv['Step']]
    y = [y for y in csv['Value']]
    X[i] = x
    Y[i] = y

# Defining which paths relate to which plot

auc_files = glob.glob('Trained models/*AUC.csv')
loss_files = glob.glob('Trained models/*loss.csv')

# Building the plot

fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle('Training and validation measures')
fig.tight_layout()

ax1.plot(X.get(auc_files[0]),Y.get(auc_files[0]), label = 'Training')
ax1.plot(X.get(auc_files[1]), Y.get(auc_files[1]), label = 'Validation')
ax1.legend()
ax1.set_title('AUC')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('AUC score')


ax2.plot(X.get(loss_files[0]), Y.get(loss_files[0]), label = 'Training')
ax2.plot(X.get(loss_files[1]), Y.get(loss_files[1]), label = 'Validation')
ax2.legend(loc = 'upper right')
ax2.set_title('Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss value')

fig.savefig('Visualizations/Tensorboard_viz.jpg', bbox_inches = 'tight')
fig.show()

#%% Loading best model and getting model summary

model_path = r'Trained models/Multiclass-deep/Multiclass-deep/dropout_0.4/model'

best_model = evaluation_functions.load_model(model_path)
print(best_model.summary())

# Getting descriptive statistics for the model

file_path = r'Trained models/Multiclass-deep/Multiclass-deep/dropout_0.4/eval_files/test_df.csv'

df = pd.read_csv(file_path)
description = df.describe()


#%% Drawing ROC curve for the best model - against human baseline and Amy Jang model performance

# Loading Amy Jang model and creating a dataframe

model_path = 'Kaggle_models/Amy_Jang.h5'
model = evaluation_functions.load_model(model_path)

eval_df, eval_im_df = evaluation_functions.Zadaulek_eval(model, X_eval, idx, 0)
pred = [x for x in eval_im_df['prediction']]

amy_df, auc = evaluation_functions.ROC_eval(pred, y_eval)

print(f'AUC score for the Amy_Jang model: {auc}')

# Loading the evaluation data performance and creating a Dataframe containing the ROC analysis

file_path = r'Trained models/Multiclass-deep/Multiclass-deep/dropout_0.4/eval_files/eval_target.csv'
eval_df = pd.read_csv(file_path)
pred = [x for x in eval_df['prediction']]
df, auc = evaluation_functions.ROC_eval(pred, y_eval)

print(f'AUC score for the best performing model: {auc}')

# Plotting both curves along with the average human performance

plt.plot(df['False positive rate'],df['True positive rate'], c = 'blue', label = 'CNN Baseline')
plt.plot(amy_df['False positive rate'], amy_df['True positive rate'], c = 'green', label = 'Amy Jang model')
plt.scatter(x_avg, y_avg, c = 'blue', label = 'Human baseline')
plt.legend(loc = 'lower right')
plt.title('CNN ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('Visualizations/CNN_ROC.jpg')
plt.show()


