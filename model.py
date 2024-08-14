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
import argparse
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from imgaug import augmenters as iaa
import cv2
import random
import os
import ntpath


def load_data(training_data_file_fullpath):
    '''
    Load the data from the full path of the training data file
    '''
    columns = ['center', 'left', 'right',
               'steering', 'throttle', 'reverse', 'speed']
    df = pd.read_csv(training_data_file_fullpath, names=columns)
    pd.set_option('display.max_colwidth', None)
    return df

def get_tail(path):
    '''
    Get the tail of the path
    '''
    remain, tail = ntpath.split(path)
    return tail


def preprocess_paths(df):
    '''
    Preprocess the paths in the data frame
    '''
    df['center'] = df['center'].apply(get_tail)
    df['left'] = df['left'].apply(get_tail)
    df['right'] = df['right'].apply(get_tail)
    return df


def balance_dataset(df, num_bins=25, sample_threshold=400):
    '''
    Balance the dataset for the steering data
    num_bins: number of bins that the steering data will be divided into
    sample_threshold: the threshold of the number of samples in each bin
    '''
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
    return df


def load_img_steering(dataset_directory, df):
    '''
    Load the image paths and steering data
    '''
    image_path = []
    steering = []
    for i in range(len(df)):
        row_data = df.iloc[i]
        center, left, right, steer = row_data[0], row_data[1], row_data[2], row_data[3]
        image_path.append(os.path.join(dataset_directory, center))
        steering.append(steer)
    return np.array(image_path), np.array(steering)

### Image augmentation functions ###
def zooming(img):
    '''
    Zoom the image
    '''
    zoom = iaa.Affine(scale=(1, 1.3))
    return zoom.augment_image(img)

def pan(img):
    '''
    Pan the image
    '''
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    return pan.augment_image(img)

def random_bright(img):
    ''' 
    Randomly change the brightness of the image
    '''
    brightness = iaa.Multiply((0.2, 1.2))
    return brightness.augment_image(img)

def image_flip(img, steering_angle):
    '''
    Flip the image and the steering angle
    '''
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

#############################################

def batch_generator(images, steering_angles, batch_size, is_training):
    '''
    Generate the batch of images and steering angles
    '''
    while True:
        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(images) - 1)
            if is_training:
                img, steering = random_augment(
                    images[random_index], steering_angles[random_index])
            else:
                img = mpltimage.imread(images[random_index])
                steering = steering_angles[random_index]
            img = img_preprocess(img)
            batch_img.append(img)
            batch_steering.append(steering)
        yield np.asarray(batch_img), np.asarray(batch_steering)

def nvidia_model(learning_rate=0.0001, dropout_rate=0.5):
    '''
    Define the Nvidia model
    '''
    model = Sequential()
    model.add(Conv2D(24, kernel_size=(5, 5), input_shape=(
        66, 200, 3), activation='elu', strides=2))
    model.add(Conv2D(36, kernel_size=(5, 5), activation='elu', strides=2))
    model.add(Conv2D(48, kernel_size=(5, 5), activation='elu', strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    adam = Adam(learning_rate=learning_rate)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=adam, metrics=['accuracy'])
    return model

def create_model(learning_rate=0.0001, dropout_rate=0.3):
    return nvidia_model(learning_rate=learning_rate, dropout_rate=dropout_rate)


def plot_history(history):
    '''
    Plot the training history
    '''
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

def load_and_preprocess_images(image_paths):
    '''
    Load and preprocess the images
    '''
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (200, 66))  # Resize to the required input shape (66, 200, 3)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        images.append(img)
    return np.array(images)

def get_best_params(X_train, y_train):
    '''
    Get the best parameters for the model
    '''
    param_grid = {
        'batch_size': [32, 64, 128],
        'epochs': [10, 20],
        'learning_rate': [0.001, 0.0001],
        'dropout_rate': [0.3, 0.5],
    }

    model = KerasRegressor(build_fn=create_model,learning_rate=0.001, dropout_rate=0.5,verbose=1)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
    return grid_result.best_params_

def train_model(X_train, y_train, X_val, y_val, params, model_name):
    '''
    Train the model
    '''
    best_model = create_model(learning_rate=params['learning_rate'], 
                              dropout_rate=params['dropout_rate'])

    history = best_model.fit(
        batch_generator(X_train, y_train, params['batch_size'], True),
        steps_per_epoch=300,
        epochs=params['epochs'],
        validation_data=batch_generator(X_val, y_val, 100, False),
        validation_steps=200,
        shuffle=True,
        verbose=1
    )

    plot_history(history)

    # Save the model
    best_model.save(model_name)
    print(f"Model saved as '{model_name}'")

def train_default(X_train, y_train, X_val, y_val, model_name):
    default_params = {
        'batch_size': 64,
        'epochs': 20,
        'learning_rate': 0.0001,
        'dropout_rate': 0.5
    }
    train_model(X_train, y_train, X_val, y_val, default_params, model_name)    
# Constants
TRAINING_DATA_DIR = 'training_data'
TRAINING_DATA_FILENAME = 'driving_log.csv'
TRAINING_DATA_IMG_DIR = 'IMG'
DEFAULT_MODEL_NAME = 'model.h5'

def main():
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument('--get_best_params', action='store_true', help='Get the best parameters using GridSearchCV')
    parser.add_argument('--train_default', type=str, help='Train the model with default parameters and save with the given model name')
    parser.add_argument('--name', type=str, help='Get best parameters and train the model with the given model name')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(script_dir, TRAINING_DATA_DIR)
    training_data_file_fullpath = os.path.join(data_directory, TRAINING_DATA_FILENAME)
    img_data_directory = os.path.join(data_directory, TRAINING_DATA_IMG_DIR)

    df = load_data(training_data_file_fullpath)
    df = preprocess_paths(df)
    df = balance_dataset(df)

    image_paths, steerings = load_img_steering(img_data_directory, df)
    X_train, X_val, y_train, y_val = train_test_split(image_paths, steerings, test_size=0.2, random_state=20)
    X_train_images = load_and_preprocess_images(X_train)
    print(X_train_images.shape)
    
    if args.get_best_params:
        get_best_params(X_train_images, y_train)
    elif args.train_default:
        if not args.train_default:
            raise ValueError("Model name must be provided with --train_default")
        train_default(X_train, y_train, X_val, y_val, args.train_default)
    elif args.name:
        if not args.name:
            raise ValueError("Model name must be provided with --name")
        best_params = get_best_params(X_train_images, y_train)
        train_model(X_train, y_train, X_val, y_val, best_params, args.name)
    else:
        print("Please provide a valid command. Use --help for more information.")

if __name__ == '__main__':
    main()
