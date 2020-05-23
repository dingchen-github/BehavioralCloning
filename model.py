import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Lambda, Cropping2D
import math
import matplotlib.pyplot as plt

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    count = 0
    for row in reader:
        count += 1
        samples.append(row)
        if count == 320:
            break

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                imgBGR = cv2.imread(name)
                center_image = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

#             steering_center = float(row[3])

#             # create adjusted steering measurements for the side camera images
#             correction = 0.2 # this is a parameter to tune
#             steering_left = steering_center + correction
#             steering_right = steering_center - correction

#             # read in images from center, left and right cameras
#             path = "..." # fill in the path to your training IMG directory
#             img_center = process_image(np.asarray(Image.open(path + row[0])))
#             img_left = process_image(np.asarray(Image.open(path + row[1])))
#             img_right = process_image(np.asarray(Image.open(path + row[2])))

#             # add images and angles to data set
#             car_images.extend(img_center, img_left, img_right)
#             steering_angles.extend(steering_center, steering_left, steering_right)

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
# model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col), output_shape=(ch, row, col)))
model.add(Cropping2D(cropping = ((70,25),(0,0)))) # crop top 70 and bottom 25 pixels, left 0 and right 0
model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation=None))
model.summary()
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

history_object = model.fit_generator(train_generator, steps_per_epoch = math.ceil(len(train_samples)/batch_size), validation_data = validation_generator,
    nb_val_samples = math.ceil(len(validation_samples)/batch_size), nb_epoch=5, verbose=1)
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
plt.show()
