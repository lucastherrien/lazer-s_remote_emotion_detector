from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPool2D, MaxPool1D, Flatten, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import sklearn.exceptions as ske
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import warnings
import datetime

img_size = 48

warnings.filterwarnings("ignore", category=ske.UndefinedMetricWarning)

#print(tf.config.list_physical_devices('GPU'))
#print(tf.config.list_logical_devices('GPU'))
rtx_3060 = '/device:GPU:0'

cam = cv2.VideoCapture(0)

while(True):
    print('Reading Data...')

    result, image = cam.read()

    #print(image.shape)

    cropped = image[int(image.shape[0]*0.45):int(image.shape[0]*0.55),
                    int(image.shape[1]*0.45):int(image.shape[1]*0.55)]

    resized = cv2.resize(image,(img_size, img_size),interpolation = cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    while result:    
        cv2.imshow("Final image", gray)
        cv2.waitKey(20)

        value = gray.astype('float32')

        value = np.asarray(value)
        value = value.flatten()
        scale = np.max(value)
        value /= scale
        mean = np.std(value)
        value -= mean
        test = tf.reshape(value,[1,img_size,img_size])

        with tf.device(rtx_3060):
            print("Generating test predictions...")
            model = load_model('best_model.h5')
            preds=model.predict(test)
            yhat = np.argmax(preds,axis=1)
            guess = yhat[0]
            #print(guess)
            target_names = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
            print(target_names[guess])
        break
    cv2.destroyWindow("Final image")
