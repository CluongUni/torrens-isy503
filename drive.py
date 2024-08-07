import socketio
import eventlet
import base64
import numpy as np
import matplotlib.image as mpltimage
import cv2
from io import BytesIO
from flask import Flask
from tensorflow import keras
#from tensorflow.keras.models import load_model
import tensorflow as tf
from keras.api.models import load_model
from PIL import Image


app = Flask(__name__)
sio = socketio.Server()
speed_limit = 10


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print(f"Raw steering angle: {steering_angle}")
    print(f"Raw throttle: {throttle}")
    print(f"Speed: {speed}")
    print("Angle: {}, Throttle: {}, Speed: {}".format(
        steering_angle, throttle, speed))
    send_command(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environment):
    print('Connected...')
    send_command(0, 0)

@sio.event
def connect_error(data):
    print("Failed to connect to the server", data)

@sio.event
def disconnect():
    print("Disconnected from the server")
    
def img_preprocess(img):
    img = img[60:130, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


def send_command(steering_angle, throttle):
    print(f"Sending: steering_angle={steering_angle}, throttle={throttle}")
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == "__main__":
    # model = load_model('model_augmented4.h5')
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
