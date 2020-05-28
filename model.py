import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Lambda, Cropping2D
import math
import matplotlib.pyplot as plt

path = './data/'
filename = 'driving_log.csv'
logfile = path + filename
samples = []
with open(logfile) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    count = 0
    for row in reader:
        count += 1
        samples.append(row)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name0 = path+'IMG/'+batch_sample[0].split('/')[-1]
                imgBGR0 = cv2.imread(name0)
                center_image = cv2.cvtColor(imgBGR0, cv2.COLOR_BGR2RGB)
                
                name1 = path+'IMG/'+batch_sample[1].split('/')[-1]
                imgBGR1 = cv2.imread(name1)
                left_image = cv2.cvtColor(imgBGR1, cv2.COLOR_BGR2RGB)
                
                name2 = path+'IMG/'+batch_sample[2].split('/')[-1]
                imgBGR2 = cv2.imread(name2)
                right_image = cv2.cvtColor(imgBGR2, cv2.COLOR_BGR2RGB)
                
                center_angle = float(batch_sample[3])
                correction = 0.2 # this is a parameter to tune
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# augmented_images, augmented_measurements = [],[]
# for image, measurement in zip(images, measurements):
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     augmented_images.append(cv2.flig(image,1)) # image_flipped = np.fliplr(image)
#     augmented_measurements.append(measurement*-1.0)
# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# crop top 70 and bottom 25 pixels, left 0 and right 0
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(36, (5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(48, (5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation=None))
model.summary()
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

history_object = model.fit_generator(
    train_generator,
    steps_per_epoch = math.ceil(len(train_samples)/batch_size),
    validation_data = validation_generator,
    nb_val_samples = math.ceil(len(validation_samples)/batch_size),
    nb_epoch=5,
    verbose=1)
model.save('./model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./loss.png')
