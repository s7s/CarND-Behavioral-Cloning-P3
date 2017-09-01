#importing some useful packages
import os
import csv
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.models import load_model
import matplotlib.pyplot as plt

#loading data from csv file
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#data spilit 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2

# using genterator to load the data
import numpy as np
import sklearn
#training generator
def tr_generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = './data/IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.cvtColor(cv2.imread(name_center), cv2.COLOR_BGR2YUV)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                #augmentation 
                #center data               
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
                
                #left data
                name_left = './data/IMG/'+batch_sample[1].split('\\')[-1]
                left_image = cv2.cvtColor(cv2.imread(name_left), cv2.COLOR_BGR2YUV)
                left_angle = float(batch_sample[3])+correction
                images.append(left_image)
                angles.append(left_angle)
                images.append(np.fliplr(left_image))
                angles.append(-left_angle)
                
                #right data
                name_right = './data/IMG/'+batch_sample[2].split('\\')[-1]
                right_image = cv2.cvtColor(cv2.imread(name_right), cv2.COLOR_BGR2YUV)
                right_angle = float(batch_sample[3])-correction
                images.append(right_image)
                angles.append(right_angle)
                images.append(np.fliplr(right_image))
                angles.append(-right_angle)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def va_generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = './data/IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.cvtColor(cv2.imread(name_center), cv2.COLOR_BGR2YUV)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                #augmentation 
                #center data               
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
                
                #left data
                name_left = './data/IMG/'+batch_sample[1].split('\\')[-1]
                left_image = cv2.cvtColor(cv2.imread(name_left), cv2.COLOR_BGR2YUV)
                left_angle = float(batch_sample[3])+correction
                images.append(left_image)
                angles.append(left_angle)
                images.append(np.fliplr(left_image))
                angles.append(-left_angle)
                
                #right data
                name_right = './data/IMG/'+batch_sample[2].split('\\')[-1]
                right_image = cv2.cvtColor(cv2.imread(name_right), cv2.COLOR_BGR2YUV)
                right_angle = float(batch_sample[3])-correction
                images.append(right_image)
                angles.append(right_angle)
                images.append(np.fliplr(right_image))
                angles.append(-right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = tr_generator(train_samples, batch_size=6)
validation_generator = va_generator(validation_samples, batch_size=6)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# trim image to only see section with road
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.))
#modified NVIDIA Architecture
model.add(Convolution2D(24,5,5,subsample=(2, 2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2, 2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2, 2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.75))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#regression proprem
model.compile(loss='mse', optimizer=keras.optimizers.Adam())
#training model
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=5,verbose=1)
#saving model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()