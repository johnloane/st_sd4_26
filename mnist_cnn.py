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
