import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import csv

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#os.chdir('train')
imgList = []

print('Creating dataset')


for dirname, _, filenames in os.walk('C:/Users/lucas/Downloads/CS229-master/CS229-master/CK+'):
    for filename in filenames:
        imgName = os.path.join(dirname, filename)
        imgList.append(imgName)

for file in imgList:
    img = Image.open(file)
    value = np.asarray(img)
    value = value.flatten()
    if 'anger' in file:
        value = np.insert(value, 0, 0)
    elif 'disgust' in file:
        value = np.insert(value, 0, 1)
    elif 'fear' in file:
        value = np.insert(value, 0, 2)
    elif 'happiness' in file:
        value = np.insert(value, 0, 3)
    elif 'sad' in file:
        value = np.insert(value, 0, 4)
    elif 'surprised' in file:
        value = np.insert(value, 0, 5)
    elif 'neutral' in file:
        value = np.insert(value, 0, 6)
    #print(value)
    #with open("data.csv", 'a') as f:
        #writer = csv.writer(f)
        #writer.writerow(value)

print('Dataset created')
