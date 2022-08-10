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
import sys
import os
import io
import png
import warnings
import datetime
import cherrypy

img_size = 48

warnings.filterwarnings("ignore", category=ske.UndefinedMetricWarning)

#print(tf.config.list_physical_devices('GPU'))
#print(tf.config.list_logical_devices('GPU'))
rtx_3060 = '/device:GPU:0'

print('Reading Data...')

#result, image = cam.read()

#print(image.shape)

class server(object):
    @cherrypy.expose
    def index(self):
        with tf.device(rtx_3060):
            return "hello world"
    @cherrypy.expose
    def process_image(self, image):
        #print(image)
        #image = cv2.imencode('.png', image)[1]
        cherrypy.response.headers['Access-Control-Allow-Origin'] = '*'
        imgFile = io.BytesIO(image.file.read())
        #print(imgFile)
        path = "image.png"
        fp = open(path, 'wb')
        imgFile.seek(0)
        fp.write(imgFile.read())
        fp.close()
        image = cv2.imread('image.png')
        cropped = image[int(image.shape[0]*0.25):int(image.shape[0]*0.75),
                        int(image.shape[1]*0.25):int(image.shape[1]*0.75)]

        resized = cv2.resize(image,(img_size, img_size),interpolation = cv2.INTER_AREA)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
           
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
            return target_names[guess]
cherrypy.quickstart(server())
