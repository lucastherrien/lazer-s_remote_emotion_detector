from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPool2D, MaxPool1D, Flatten, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam, SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sklearn.exceptions as ske
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler as ROS
import pandas as pd
import numpy as np
import warnings
import datetime
from PIL import Image
import os

img_size_x = 640
img_size_y = 490

warnings.filterwarnings("ignore")

#print(tf.config.list_physical_devices('GPU'))
#print(tf.config.list_logical_devices('GPU'))
rtx_3060 = '/device:GPU:0'
oversampler = ROS(sampling_strategy='auto')

imgList = []
data = pd.DataFrame([])
i = 0

print('Creating dataset')


for dirname, _, filenames in os.walk('C:/Users/lucas/Downloads/CS229-master/CS229-master/CK+'):
    for filename in filenames:
        imgName = os.path.join(dirname, filename)
        imgList.append(imgName)

for file in imgList:
    i = i+1
    print(f"image {i} out of {len(imgList)}")
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
    value = pd.DataFrame(value)
    data = data.append(value)

with tf.device(rtx_3060):
    # Read data
    print('Reading head of Data...')

    #data = pd.read_csv('data.csv',nrows=1,chunksize=1000)\
    
    print('Reading Data...')

    #data = pd.read_csv('data.csv',chunksize=1000)

    print('data loaded, extrapolating x and y...')
    print(data.head())
    labels = data.iloc[:,0].values.astype('int32')
    print(labels[0:5])
    X = data.iloc[:,1:].values.astype('float32')

    # convert list of labels to binary class matrix
    y = np_utils.to_categorical(labels)
    #print(test_labels)

    print('test train split...')

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # pre-processing: divide by max and substract mean
    scale = np.max(X_train)
    X_train /= scale
    X_test /= scale

    mean = np.std(X_train)
    X_train -= mean
    X_test -= mean

    #print(X_train.shape)

    #X_train,y_train = oversampler.fit_resample(X_train,y_train)
    #X_test,y_test = oversampler.fit_resample(X_test,y_test)
    
    #input_dim = X_train.shape[1]
    nb_classes = y_train.shape[1]
    #print(nb_classes)
    #print(X_train.shape)

    X_train = tf.reshape(X_train,[X_train.shape[0],img_size_x,img_size_y])
    X_test = tf.reshape(X_test,[X_test.shape[0],img_size_x,img_size_y])

    #print(X_train.shape)
    #print(X_train[0:5])

    # Here's a Deep Dumb MLP (DDMLP)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(640,490,1)))
    model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
        
    model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) 
    model.add(Dense(256,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
        
    model.add(Dense(512,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy',
                         mode='max', verbose=1, save_best_only=True)
    # we'll use categorical xent for the loss, and adam as the optimizer
    model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=0.001),
                  metrics = ['accuracy'])
    model.summary()
    print("Training...")
    history = model.fit(X_train, y_train, epochs=500,batch_size=64,
              callbacks=[es,mc],validation_data=(X_test,y_test), verbose=2)
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    fig.set_size_inches(12,4)

    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Training Accuracy vs Validation Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')

    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Training Loss vs Validation Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')

    print("Generating test predictions...")
    model = load_model('best_model.h5')
    preds=model.predict(X_test)
    yhat = np.argmax(preds,axis=1)
    test_labels = np.argmax(y_test,axis=1)

    def write_preds(preds, fname):
        pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)
    now = str(datetime.datetime.today()).replace('.','_')
    now = now.replace(':','_')
    write_preds(yhat,"preds "+now+".csv")

    
    target_names = ['Angry','Disgust','Fear','Happiness','Sad','Surprise','Neutral']
    print("Classification Report:")
    print(classification_report(test_labels,yhat,target_names=target_names))

    plt.show()
