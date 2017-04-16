from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D, Dropout
from keras.callbacks import EarlyStopping
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

samples = []
with open('./driving_log_1.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('./driving_log_2.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('./driving_log_3.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('./driving_log_4.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
with open('./driving_log_5.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

batch_size = 32
correction = 0.3

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates

        # Disabled shuffling of all samples

        # sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            # Batches are shuffled not the samples
            sklearn.utils.shuffle(batch_samples)

            images = []
            angles = []
            for batch_sample in batch_samples:

                center_image = cv2.imread('./'+batch_sample[0])
                left_image = cv2.imread('./'+batch_sample[1])
                right_image = cv2.imread('./'+batch_sample[2])

                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])

                images.extend([cv2.flip(center_image, 1), cv2.flip(left_image, 1), cv2.flip(right_image, 1)])
                angles.extend([-center_angle, -left_angle, -right_angle])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield  X_train, y_train

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

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

model.compile(loss='mse', optimizer='adam')

model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_samples)/batch_size,
    validation_data=validation_generator,
    validation_steps=len(validation_samples)/batch_size,
    epochs=2,
    callbacks=[
        # Added early stopping to prevent the network from getting worse
        EarlyStopping(
            monitor='val_loss',
            mode='min'
        )
    ]
)

model.save('model.h5')
