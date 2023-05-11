import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import csv

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#os.chdir('train')
imgList = []
two_datasets = True

print('Creating train dataset')

if(two_datasets):
    print('Dataset 1/2')

for dirname, _, filenames in os.walk('C:/Users/lucas/Downloads/archive/train'):
    for filename in filenames:
        imgName = os.path.join(dirname, filename)
        imgList.append(imgName)

for file in imgList:
    img = Image.open(file)
    value = np.asarray(img)
    value = value.flatten()
    if 'angry' in file:
        value = np.insert(value, 0, 0)
    elif 'disgusted' in file:
        value = np.insert(value, 0, 1)
    elif 'fearful' in file:
        value = np.insert(value, 0, 2)
    elif 'happy' in file:
        value = np.insert(value, 0, 3)
    elif 'sad' in file:
        value = np.insert(value, 0, 4)
    elif 'surprised' in file:
        value = np.insert(value, 0, 5)
    elif 'neutral' in file:
        value = np.insert(value, 0, 6)
    #print(value)
    with open("train.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)

if(two_datasets):
    print('Dataset 2/2')

    train2 = pd.read_csv('train2/train.csv')
    emotions = train2['emotion']
    images = train2['pixels']
    totalImages = images.shape[0]

    for i in range(totalImages):
        str_img = images[i]
        img_array_str = str_img.split(' ')
        value = np.asarray(img_array_str, dtype=np.uint8).reshape(48,48)
        value = value.flatten()
        value = np.insert(value, 0, emotions[i])
        with open("train.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)

print('Creating test dataset')

imgList = []

for dirname, _, filenames in os.walk('C:/Users/lucas/Downloads/archive/test'):
    for filename in filenames:
        imgName = os.path.join(dirname, filename)
        imgList.append(imgName)

for file in imgList:
    img = Image.open(file)
    value = np.asarray(img)
    value = value.flatten()
    if 'angry' in file:
        value = np.insert(value, 0, 0)
    elif 'disgusted' in file:
        value = np.insert(value, 0, 1)
    elif 'fearful' in file:
        value = np.insert(value, 0, 2)
    elif 'happy' in file:
        value = np.insert(value, 0, 3)
    elif 'sad' in file:
        value = np.insert(value, 0, 4)
    elif 'surprised' in file:
        value = np.insert(value, 0, 5)
    elif 'neutral' in file:
        value = np.insert(value, 0, 6)
    #print(value)
    with open("test.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)

print('Datasets created')
