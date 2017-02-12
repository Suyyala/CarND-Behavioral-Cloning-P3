import numpy as np
import warnings

from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from skimage import data, io, filters


# the car should stay in the center of the road as much as possible
# if the car veers off to the side, it should recover back to center
# driving counter-clockwise can help the model generalize
# flipping the images is a quick way to augment the data
# collecting data from the second track can also help generalize the model
# we want to avoid overfitting or underfitting when training the model
# knowing when to stop collecting more data


def model_dave2_net():
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    ch, row, col = 3, 80, 160  # Trimmed image format
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
    model.add(Convolution2D(24, 5, 5, input_shape=(row, col, ch), activation='relu'))
    model.add(Convolution2D(36, 5, 5, activation='relu'))
    model.add(Convolution2D(48, 5, 5, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    #model.add(Dense(1024))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    return model


def image_process(image):
    # NOTE: img[y: y + h, x: x + w]    
    mod_image = image[80:,:,:]
    mod_image = cv2.resize(mod_image, (160, 80))
    return mod_image

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Used as a reference pointer so code always loops back around
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_train = np.zeros((batch_size, 80, 160, 3))
            y_train = np.zeros((batch_size, 1))
            images = []
            angles = []
            is_mirror = False
            for batch_sample in batch_samples:
                name = '/home/suyyala/IMG/'+batch_sample[0].split('/')[-1]
                #name = 'IMG/'+batch_sample[0].split('/')[-1]
                if batch_sample[0].split('/')[-1].startswith('mirror_') is True:
                    is_mirror = True
                
                
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                center_image = image_process(center_image)
                if(is_mirror is True):
                    center_image = cv2.flip(center_image,flipCode=1) 
                    center_angle = -center_angle
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train =  np.array(angles)
            yield shuffle(X_train, y_train)

samples = []
with open('/home/suyyala/driving_log.csv') as csvfile:
#with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader, None)
    for line in reader:
        samples.append(line)
        line_copy = line.copy()
        line_copy_list = line_copy[0].split('/')
        line_copy_list[-1] = 'mirror_' + line_copy_list[-1]
        line_copy[0] = '/'.join(line_copy_list)

        

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(validation_samples[0])

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model =  model_dave2_net()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=3)
# always save your weights after training or during training
model.save('model.h5')  

	
