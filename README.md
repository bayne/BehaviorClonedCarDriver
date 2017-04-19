# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Model Architecture and Training Strategy

### Model

The network architecture is a slightly modified model [provided by Nvidia](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) that is used for a self-driving car.

**Nvidia's Model**
![cnn-architecture-624x890](https://cloud.githubusercontent.com/assets/712014/25165631/51bb63dc-248c-11e7-8fd6-ebdbf7549a36.png)

The primary differences in my model are:
- the removal of the 10 neuron fully connected layer
- an addition of a dropout layer
- some additional pre-processing steps

The keras framework allows us to describe the model pretty succinctly:

```python
model = Sequential()

# trim image to only see section with road
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# Normalize and center the data around 0
model.add(Lambda(lambda x: x/127.5 - 1.))

# Nvidia's architecture
model.add(Conv2D(24, (5, 5), padding='same', activation='relu', strides=(2, 2)))
model.add(Conv2D(36, (5, 5), padding='same', activation='relu', strides=(2, 2)))
model.add(Conv2D(48, (5, 5), padding='same', activation='relu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(100))

# Added dropout to prevent overfitting
model.add(Dropout(0.6))

model.add(Dense(50))
model.add(Dense(1))
```

#### Pre-processing

For pre-processing the images, I added a layer to normalize the images (for better numeric stability) and cropping to reduce the noise from the cameras.

#### Overfitting prevention

To prevent overfitting, a dropout layer was added. Keras abstracts the validation mechanism which also provides protection from overfitting.

### Appropriate training data

Since this is an end-to-end behavior cloning system, the training data and how that training data is handled determines if the network will provided the desired result.

#### Shuffling

One of the key changes I made that yielded the best results was changing how shuffling was working. Initially the shuffling of training data was done on a per frame basis, however, I intuited that there was valuable information in the order of the frames. I changed the shuffling to shuffle batches of frames rather than the individual frames themselves. I believe this provided the network extra information on how the steering angle changed as a group of sequential frames changed.

To still get the benefit of shuffling to prevent overfitting, the shuffling was updated to still shuffle groups of frames.

#### Camera Angles

![image](https://cloud.githubusercontent.com/assets/712014/25166367/f3e7a9ec-248f-11e7-83db-319b8a2555cf.png)

The simulator also provides multiple views on the car that can be used to provided additional training data for the steering angle. Since the driving mode only provides one viewport, a new hyperparameter had to be introduced that represents a steering offset that would be expected for the given viewports.

#### Less than ideal training

Initially it seems to make sense to just provide training data that shows perfect runs of the track. When you only provide the ideal path, the network is unable to react to conditions that are less than ideal. By providing training data of a driver that is swerving inside the lane, the network is given better ranges of steering angles along with frames that show the conditions in which the car will fall off the road if corrective action isn't taken.

#### Data augmentation

For additional more varied training data, a simple augmentation is to flip the images and flip the associated steering angle. This doubles the amount of data provided to the network.
