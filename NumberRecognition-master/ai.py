# Importing necesary library
import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import json

# Loading MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Loading pre-built model
model = tf.keras.models.load_model('ai.h5')

# Preprocessing image
def preprocess(image):
    image = tf.constant(image)
    return np.array([image])

def display(image):
    plt.imshow(image,cmap='binary')
    plt.savefig('sample.png')

# Selecting single image
one_data = preprocess(x_test[0])
display(x_test[0])

# Predicting single image
prediction_conf = model.predict(one_data)
prediction = np.argmax(prediction_conf)

# Exporting into json
json_data = {"prediction" : int(prediction)}
with open('prediction.json','w') as output:
    json.dump(json_data,output)