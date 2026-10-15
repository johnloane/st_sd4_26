import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import requests
from PIL import Image
import cv2

import random

np.random.seed(0)

num_pixels = 784
num_classes = 10

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape)
    print(X_test.shape)
    check_data_shape(X_train, y_train, X_test, y_test)
    print_five_each_number(X_train, y_train)
    X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test)
    
 
 def preprocess(X_train, y_train, X_test, y_test):
     y_train = to_categorical(y_train, num_classes)
     y_test = to_categorical(y_test, num_classes)
    
def print_five_each_number(X_train, y_train):
    cols = 5
    number_of_samples = []
    fig, axs = plt.subplots(nrows = num_classes, ncols=cols, figsize=(5, 10))
    fig.tight_layout()
    for i in range(cols):
        for j in range(num_classes):
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected-1)), :, :], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j))
                number_of_samples.append(len(x_selected))
    graph_sample_distribution(num_classes, number_of_samples)
    plt.show()
    
    
def graph_sample_distribution(num_classes, number_of_samples):
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), number_of_samples)
    plt.title("Distribution of the training set")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
    
    
def check_data_shape(X_train, y_train, X_test, y_test):
    assert(X_train.shape[0] == y_train.shape[0]), "The number of images in the training set is not equal to the number of labels"
    assert(X_test.shape[0] == y_test.shape[0]), "The number of images in the test set is not equal to the number of labels"
    assert(X_train.shape[1:] == (28, 28)), "The dimensions of the images in the training set are not 28 by 28"
    assert(X_test.shape[1:] == (28, 28)), "The dimensions of the images in the test set are not 28 by 28"
    
    
    
if __name__ == "__main__":
    main()