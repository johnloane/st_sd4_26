import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model

import random
import requests
from PIL import Image
import cv2

np.random.seed(0)

num_pixels = 784
num_classes = 10

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape)
    print(X_test.shape)
    check_data_shape(X_train, y_train, X_test, y_test)
    X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test)
    model = leNet_model()
    print(model.summary())
    train_and_evaluate(model, X_train, y_train, X_test, y_test)
    
    
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    h = model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=400, verbose=1, shuffle=1)
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.show()
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.show()
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score: ', score[0])
    print('Test accuracy: ', score[1])
    
    
    
def leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5,5), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
    
    
def preprocess(X_train, y_train, X_test, y_test):
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    X_train = X_train/255
    X_test = X_test/255
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    return X_train, y_train, X_test, y_test
    
    
def check_data_shape(X_train, y_train, X_test, y_test):
    assert(X_train.shape[0] == y_train.shape[0]), "The number of images in the training set is not equal to the number of labels"
    assert(X_test.shape[0] == y_test.shape[0]), "The number of images in the test set is not equal to the number of labels"
    assert(X_train.shape[1:] == (28, 28)), "The dimensions of the images in the training set are not 28 by 28"
    assert(X_test.shape[1:] == (28, 28)), "The dimensions of the images in the test set are not 28 by 28"
    
    
if __name__ == "__main__":
    main()
