import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpltimage
import tensorflow as tf
from tensorflow import keras
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from keras.api.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.api.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import random
import os
import ntpath

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the data directory relative to the script directory
datadir = os.path.join(script_dir, 'training_data')

# Read the CSV file
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
df = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names=columns)
pd.set_option('display.max_colwidth', None)

# Function to get the tail of a path
def get_tail(path):
    remain, tail = ntpath.split(path)
    return tail

# Apply get_tail function to center, left, and right columns
df['center'] = df['center'].apply(get_tail)
df['left'] = df['left'].apply(get_tail)
df['right'] = df['right'].apply(get_tail)

# Balance the dataset
num_bins = 25
sample_threshold = 400
hist, bins = np.histogram(df['steering'], num_bins)
center = (bins[:-1] + bins[1:]) / 2

remove_list = []
for j in range(num_bins):
    lst = []
    for i in range(len(df['steering'])):
        if bins[j] <= df['steering'][i] <= bins[j+1]:
            lst.append(i)
    lst = shuffle(lst)
    lst = lst[sample_threshold:]
    remove_list.extend(lst)

df.drop(remove_list, inplace=True)

# Load images and steering data
def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(df)):
        row_data = df.iloc[i]
        center, left, right, steer = row_data[0], row_data[1], row_data[2], row_data[3]
        image_path.append(os.path.join(datadir, center))
        steering.append(steer)
    return np.array(image_path), np.array(steering)

image_paths, steerings = load_img_steering(os.path.join(datadir, 'IMG'), df)
X_train, X_val, y_train, y_val = train_test_split(image_paths, steerings, test_size=0.2, random_state=20)

# Image augmentation functions
def zooming(img):
    zoom = iaa.Affine(scale=(1, 1.3))
    return zoom.augment_image(img)

def pan(img):
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    return pan.augment_image(img)

def random_bright(img):
    brightness = iaa.Multiply((0.2, 1.2))
    return brightness.augment_image(img)

def image_flip(img, steering_angle):
    img = cv2.flip(img, 1)
    steering_angle = -steering_angle
    return img, steering_angle

def random_augment(img_path, steering_angle):
    img = mpltimage.imread(img_path)
    if np.random.rand() < 0.5:
        img = pan(img)
    if np.random.rand() < 0.5:
        img = zooming(img)
    if np.random.rand() < 0.5:
        img, steering_angle = image_flip(img, steering_angle)
    if np.random.rand() < 0.5:
        img = random_bright(img)
    return img, steering_angle

def img_preprocess(img):
    img = img[60:130, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

# Batch generator
def batch_generator(images, steering_angles, batch_size, is_training):
    while True:
        batch_img = []
        batch_steering = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(images) - 1)
            if is_training:
                img, steering = random_augment(images[random_index], steering_angles[random_index])
            else:
                img = mpltimage.imread(images[random_index])
                steering = steering_angles[random_index]
            img = img_preprocess(img)
            batch_img.append(img)
            batch_steering.append(steering)
        yield np.asarray(batch_img), np.asarray(batch_steering)

# NVIDIA model
def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, kernel_size=(5, 5), input_shape=(66, 200, 3), activation='elu', strides=2))
    model.add(Conv2D(36, kernel_size=(5, 5), activation='elu', strides=2))
    model.add(Conv2D(48, kernel_size=(5, 5), activation='elu', strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    
    adam = Adam(learning_rate=0.0001)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=adam, metrics=['accuracy'])
    return model

# Create and train the model
model = nvidia_model()
model.summary()

history = model.fit(
    batch_generator(X_train, y_train, 100, True),
    steps_per_epoch=300,
    epochs=10,
    validation_data=batch_generator(X_val, y_val, 100, False),
    validation_steps=200,
    shuffle=True,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.show()

# Save the model
model.save('model.h5')
print("Model saved as 'model.h5'")