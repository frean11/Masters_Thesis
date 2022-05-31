# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:19:37 2022

@author: frede
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import kaggle
from zipfile import ZipFile
import matplotlib.pyplot as plt
from collections import Counter
import random

from PIL import Image
import PIL

import shutil
from glob import glob
import os
os.chdir(os.path.dirname(__file__))

# Relevant paths

path = 'SIIM-ISIC'

# Creating folder

if not os.path.isdir(path):
    os.mkdir(path)
    
# If the folder is empty, we want to download some content into it

if len(os.listdir(path)) == 0:
    os.chdir(path)
    
    # Downloading the train.csv
    !kaggle competitions download -f train.csv siim-isic-melanoma-classification
    try: 
        with ZipFile('train.csv.zip', 'r') as zip:
            zip.extractall()
        os.remove('train.csv.zip')
    except:
        pass
        
    # Downloading test.csv
    !kaggle competitions download -f test.csv siim-isic-melanoma-classification
    try:
        with ZipFile('test.csv.zip', 'r') as zip:
            zip.extractall()
        os.remove('test.csv.zip')
    except:
        pass


"""
The Kaggle API did not allow for downloading specific folders (as I only wanted the test and training folders with .jpg files)
I instead downloaded the folder "jpeg" from https://www.kaggle.com/c/siim-isic-melanoma-classification/data?select=jpeg manually.
The download was done on March 7th, 2022.
The folder was downloaded into the SIIM-ISIC folder created in this script and called "Images".
Pre-processing using Python continues in the next cell
"""

#%%

# Unpacking the zip files

if 'Images.zip' in os.listdir(path):
    with ZipFile(path+'/Images.zip', 'r') as zip:
        zip.extractall()
    os.remove(path+'/Images.zip')

# Function for handling the reshaping of images

def image_resize(path):
    image = Image.open(path)
    image_resized = image.resize((768,512))
    image_resized.save(path)

# Reshaping test images first

TEST_FILES = os.listdir(path+'/test')
TEST_FILE = TEST_FILES[-1]

TEST_IM = Image.open(path+'/test/'+TEST_FILE)
if TEST_IM.size != (768,512):
    print(TEST_IM.size)
    print('Time to go to work')   
    for i in tqdm(TEST_FILES):
        image_resize('test/'+i)
else:
    print('Looks like the images are correct size')

# Same procedure for training images

TRAIN_FILES = os.listdir(path+'/train')
TRAIN_FILE = TRAIN_FILES[-1]

TRAIN_IM = Image.open(path+'/train/'+TRAIN_FILE)
if TRAIN_IM.size != (768,512):
    print(TRAIN_IM.size)
    print('Time to go to work')   
    for i in tqdm(TRAIN_FILES):
        image_resize('train/'+i)
else:
    print('Looks like the images are correct size')


# How many postive instances in training set?

train_data = pd.read_csv('SIIM-ISIC/train.csv')

print(f"There are {train_data['target'].sum()/len(train_data)*100} % positive instances in the dataset")

"""
Only 1.76 % positive instances, giving a no information model of 98.24 %.
It will be difficult to train a model to spot so few positives.
Therefore, I decide to also download and include the 2019 data set, it has a much higher positive rate.
"""

#%% Importing the 2019 Kaggle dataset (including the 2018 HAM10000 data) for additional positive instances

"""
This 2019 dataset was downloaded manually through the official ISIC website:
    https://challenge.isic-archive.com/data/
All files are stored in the 'SIIM-ISIC' folder.
The following section will unpack and prepare the images.
"""

# First unpacking the images

if '2019_images.zip' in os.listdir(path):
    if not os.path.isdir(path+'/2019_images'):
        os.mkdir(path+'/2019_images')
    with ZipFile(path+'/2019_images.zip', 'r') as zip:
        zip.extractall(path+'/2019_images')
    os.remove(path+'/2019_images.zip')


# Removing files which are not images

full_path = 'SIIM-ISIC/2019_images/ISIC_2019_Training_Input'

FILES = os.listdir(full_path)
for i in tqdm(FILES):
    if not '.jpg' in i:
        os.remove(full_path+'/'+i)

# And then rescaling to 768x512

try:
    FILES = os.listdir(full_path)
    FILE = FILES[-1]
    
    TEST_IM = Image.open(full_path+'/'+FILE)
    if TEST_IM.size != (768,512):
        print(TEST_IM.size)
        print('Time to go to work')   
        for i in tqdm(FILES):
            image_resize(full_path+'/'+i)
    else:
        print('Looks like the images are correct size')
    TEST_IM.close()
except:
    print('Does the 2019_images directory exist?')

# Doing some descriptive analysis before moving on.

path_ = 'SIIM-ISIC/Descriptives'

try:
    os.mkdir(path_)
except:
    pass

# Any overlap between the image ID's in the datasets?

data_2019 = pd.read_csv(path+'/2019_truth.csv')

for i in data_2019['image']:
    if i in train_data['image_name']:
        raise Exception('There is an id overlap between the 2019 and the 2020 images')

# Descriptives 

train_counts = train_data['diagnosis'].value_counts()
train_counts.to_csv(path_+'/ISIC_2020_diagnosis_descriptives.csv')
plot = train_counts.plot(kind = 'bar',
                         title = '2020 ISIC diagnosis distribution',
                         xlabel = 'lesion type',
                         ylabel = 'count').get_figure()
plot.savefig(path_+'/2020_diagnosis_descriptives.jpg')

# First creating a diagnosis column in the 2019 overview similar to the one in the 2020 overview

if not 'diagnosis' in data_2019.columns:
    data_2019['diagnosis'] = data_2019.loc[:, data_2019.columns != 'image'].idxmax(axis=1)
data_2019_counts = data_2019['diagnosis'].value_counts()
data_2019_counts.to_csv(path_+'/ISIC_2019_diagnosis_descriptives.csv')
plot = data_2019_counts.plot(kind = 'bar', 
                       title = '2019 ISIC diagnosis distribution',
                       xlabel = 'lesion type',
                       ylabel = 'count').get_figure()
plot.savefig(path_+'/2019_diagnosis_descriptive.jpg')

# Remapping the dataframes

remap_train = {'nevus' : 'NV',
               'melanoma' : 'MEL',
               'seborrheic keratosis' : 'BKL',
               'lichenoid keratosis' : 'BKL',
               'solar lentigo' : 'BKL',
               'lentigo NOS' : 'BKL',
               'cafe-au-lait macule' : 'Unknown',
               'atypical melanocytic proliferation' : 'Unknown',
               'unknown' : 'Unknown'}

train_replaced = train_data.replace({'diagnosis' : remap_train})


# Moving the 2019 images into the training folder

full_path = 'SIIM-ISIC/2019_images/ISIC_2019_Training_Input'
train_path = 'SIIM-ISIC/train'
val_path = 'SIIM-ISIC/validation'

if os.path.isdir(full_path):
    files = os.listdir(full_path)
    for i in tqdm(files):
        shutil.move(full_path+'/'+i, train_path+'/'+i)
    assert len(os.listdir(full_path)) == 0 # Feel free to delete the 2019_images folder, it will not be used anymore


# Making validation folder and sampling 15% of the images into it

if not os.path.isdir(val_path):
    os.mkdir(val_path)

if len(os.listdir(val_path)) == 0:
    print("Validation folder empty, let's put some pictures into it")
    IMAGE_LIST = os.listdir(train_path)
    random.seed(1337)
    VAL_LIST = random.sample(IMAGE_LIST, int(len(IMAGE_LIST)*0.15))
    for i in tqdm(VAL_LIST):
        shutil.move(train_path+'/'+i, val_path+'/'+i)

# Generating the appropriate subfolders and moving files into them

categories = set(train_replaced['diagnosis']).union(set(data_2019['diagnosis']))

for i in categories:
    if not os.path.isdir(train_path+'/'+i):
        os.mkdir(train_path+'/'+i)
    if not os.path.isdir(val_path+'/'+i):
        os.mkdir(val_path+'/'+i)


train_list = os.listdir(train_path)
val_list = os.listdir(val_path)

for index, row in tqdm(train_replaced.iterrows()):
    img_name = row['image_name']+'.jpg'
    img_cat = row['diagnosis']
    
    if img_name in train_list:
        shutil.move(train_path+'/'+img_name, train_path+'/'+img_cat+'/'+img_name)
    
    elif img_name in val_list:
        shutil.move(val_path+'/'+img_name, val_path+'/'+img_cat+'/'+img_name)
    
    else:
        continue

for index, row in tqdm(data_2019.iterrows()):
    img_name = row['image']+'.jpg'
    img_cat = row['diagnosis']
    
    if img_name in train_list:
        shutil.move(train_path+'/'+img_name, train_path+'/'+img_cat+'/'+img_name)
    
    elif img_name in val_list:
        shutil.move(val_path+'/'+img_name, val_path+'/'+img_cat+'/'+img_name)
    
    else:
        continue

# Last step is making sure there is exactly one folder in the testing folder

TEST_PATH = 'SIIM-ISIC/test'

if not os.path.isdir(TEST_PATH+'/test'):
    os.mkdir(TEST_PATH+'/test')

TEST_IMAGES = os.listdir(TEST_PATH)

for i in tqdm(TEST_IMAGES):
    if '.jpg' in i:
        shutil.move(TEST_PATH+'/'+i,TEST_PATH+'/test/'+i)

"""
And this concludes the image import and cleaning.
The training and validation images are now divided into the appropriate folder structure to utilize the Keras ImageDataGenerator.
"""

#%%

"""
Update March 28th. It turned out that a structure with binary image labelling could also be a good idea.
The following script creates a new file folder system which is ready for binary classification.
"""

# Setting up the old and new directory

src = 'SIIM-ISIC'
dest = 'SIIM-ISIC_binary'

# Creating the target folders

try:
    os.mkdir(dest)
    os.mkdir(dest+'/train')
    os.mkdir(dest+'/train/malignant')
    os.mkdir(dest+'/train/benign')
    os.mkdir(dest+'/validation')
    os.mkdir(dest+'/validation/malignant')
    os.mkdir(dest+'/validation/benign')
except:
    pass

# Moving the malignant files first, from the "MEL" category

train_malignant = os.listdir(src+'/train/MEL')
for i in train_malignant:
    shutil.copy(src+'/train/MEL/'+i, dest+'/train/malignant/'+i)

# Then moving the rest of the images one by one

train_folders = os.listdir(src+'/train')

for j in tqdm(train_folders):
    if j != 'MEL':
        files = os.listdir(src+'/train/'+j)
        for file in files:
            shutil.copy(src+'/train/'+j+'/'+file, dest+'/train/benign/'+file)

# Repeat above process for the validation images

val_malignant = os.listdir(src+'/validation/MEL')
for i in val_malignant:
    shutil.copy(src+'/validation/MEL/'+i, dest+'/validation/malignant/'+i)

val_folders = os.listdir(src+'/validation')

for j in tqdm(val_folders):
    if j != 'MEL':
        files = os.listdir(src+'/validation/'+j)
        for file in files:
            shutil.copy(src+'/validation/'+j+'/'+file, dest+'/validation/benign/'+file)

