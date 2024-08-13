import argparse
import socketio
import eventlet
import base64
import numpy as np
import matplotlib.image as mpltimage
import cv2
from io import BytesIO
from flask import Flask
from tensorflow import keras
import tensorflow as tf
from keras.api.models import load_model
from PIL import Image


app = Flask(__name__)
sio = socketio.Server()
speed_limit = 10

def img_preprocess(img):
    img = img[60:130, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

def preprocess_image(image_data):
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = np.asarray(image)
    return img_preprocess(image)

def calculate_throttle(speed):
    return 1.0 - speed / speed_limit

def log_telemetry(steering_angle, throttle, speed):
    print(f"Raw steering angle: {steering_angle}")
    print(f"Raw throttle: {throttle}")
    print(f"Speed: {speed}")
    print(f"Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}")

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = calculate_throttle(speed)
    log_telemetry(steering_angle, throttle, speed)
    send_command(steering_angle, throttle)

def send_command(steering_angle, throttle):
    print(f"Sending: steering_angle={steering_angle}, throttle={throttle}")
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

@sio.on('connect')
def connect(sid, environment):
    print('Connected...')
    send_command(0, 0)

@sio.event
def connect_error(data):
    print(f"Failed to connect to the server - data {data}")

@sio.event
def disconnect(sid):
    print(f"Disconnected from the server - sid {sid} ")
    

DEFAULT_MODEL = 'model.h5'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Self-driving car model server')
    parser.add_argument('--model', type=str, help='model name')
    args = parser.parse_args()

    model_path = args.model if args.model else DEFAULT_MODEL
    model = load_model(model_path)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
