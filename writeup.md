# **Behavioral Cloning**

[//]: # (Image References)

[image1]: ./model.png "model"
[image2]: ./loss.png "loss"
[image3]: ./run2.png "run2"
[image4]: ./run4.png "run4"
[image5]: ./run6.png "run6"

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator, the drive.py file and the model file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model comes from the NVIDIA paper [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) as one of the recommendations from the course. This paper describes exactly what we are doing in this project: create and train a model to output the steering angles.

My model
* takes in RGB images,
* normalizes them,
* chops from them the top 70 and bottom 25 pixels of no useful information,
* consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64,
* three fully connected layers,
* and one output layer.

My model uses RELU activation function to introduce nonlinearity.

```
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
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
```

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting.

```
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```

From the losses we can see that this model does not have an overfitting problem. So no dropout is introduced in the model.

![alt text][image2]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

```
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
```

#### 4. Appropriate training data

For the "easy" track, I simply used the training data provided by Udacity. For the "hard" track, I used center lane driving first. Then I found out that the car did not perform well in sharp curves, so I collected extra training data in curves.

For details about how I created the training data, see the next section.

#### 5. Data generator

In order to avoid excessive memory usage, a data generator was deployed, which yields the shuffled training data, as can be seen in the function `def generator(samples, batch_size=32)`.

---

### Data collection and driving performance

#### 1. First try with Udacity training data and center camera images

I trained the model first with only Udacity training data on the "easy" track and only center camera images. Then I let the car drive autonomously in the simulator. On the "easy" track, the cars drives surprisely well, as can be seen in [run1.mp4](https://github.com/dingchen-github/BehavioralCloning/blob/master/run1.mp4).

However, on the "hard" track, the car "crashed" into a hill on the first curve.

![alt text][image3]

#### 2. Second try with my training data and center camera images

I drove a full lap on the "hard" track and fed center camera images to the model. This time the car drove autonomously for more than 30 seconds before hitting the poles.

![alt text][image4]

#### 3. Third try with center, left and right camera images

My guess is that the car needed side camera images to correct its steering angles, so I fed center, left and right camera images to the model. This time the car drove autonomously for more than 90 seconds before hitting the poles in a sharp curve.

![alt text][image5]

#### 4. Fourth try with extra curve training data

The model needed more training for sharp curves, so I drove the car over several curves and fed the data to the model. This time the car drove autonomously for the whole lap, as can be seen in [video_challenge.mp4](https://github.com/dingchen-github/BehavioralCloning/blob/master/video_challenge.mp4).

And of cource, the car drove perfectly well on the "easy" track, as can be seen in [video.mp4](https://github.com/dingchen-github/BehavioralCloning/blob/master/video.mp4).
