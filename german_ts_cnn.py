import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D
import pickle
import pandas as pd
import cv2
import requests
from PIL import Image

np.random.seed(0)

def main():
    num_classes = 43
    train_data, val_data, test_data = unpickle()
    X_train, y_train, X_val, y_val, X_test, y_test = check_data(train_data, val_data, test_data)
    data = pd.read_csv('german_traffic_signs/signnames.csv')
    print(data)
    num_of_each_sign = show_training_samples(num_classes, data, X_train, y_train)
    plot_sample_distribution(num_classes, num_of_each_sign)
    examine_typical_image(X_train, y_train)
    X_train, X_val, X_test = apply_preprocessing(X_train, X_val, X_test)
    examine_random_training_image_after_preprocessing(X_train)
    X_train, X_val, X_test = reshape_for_cnn(X_train, X_val, X_test)
    y_train, y_val, y_test = one_hot_encode(num_classes, y_train, y_val, y_test)
    model = leNet_model(num_classes)
    
    
def leNet_model(num_classes):
    model = Sequential()
    model.add(Conv2D(60, (5,5), input_shape=(32,32,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning=0.001, loss='categorical_crossentropy', metrics=['accuracy']))
    return model
   
 
def one_hot_encode(num_classes, y_train, y_val, y_test):
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return y_train, y_val, y_test

   
def examine_random_training_image_after_preprocessing(X_train):
    plt.imshow(X_train[np.random.randint(0, len(X_train)-1)])
    plt.axis("off")
    plt.show()
    print(X_train.shape)
    
def reshape_for_cnn(X_train, X_val, X_test):
    X_train = X_train.reshape(34799, 32, 32, 1)
    X_val = X_val.reshape(4410, 32, 32, 1)
    X_test = X_test.reshape(12630, 32, 32, 1)
    return X_train, X_val, X_test
    
    
def apply_preprocessing(X_train, X_val, X_test):
    X_train = np.array(list(map(preprocessing, X_train)))
    X_val = np.array(list(map(preprocessing, X_val)))
    X_test = np.array(list(map(preprocessing, X_test)))
    return X_train, X_val, X_test
    
    
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
    
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def equalize(img):
    # This only works for grayscale images
    img = cv2.equalizeHist(img)
    return img

def examine_typical_image(X_train, y_train):
    pre_img = X_train[500]
    plt.imshow(pre_img)
    plt.show()
    
    
def plot_sample_distribution(num_classes, num_of_each_sign):
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_each_sign)
    plt.title("Distribution of the training set")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
    
    
def show_training_samples(num_classes, data, X_train, y_train):
    num_samples = []
    cols = 5
    
    fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(10, 50))
    fig.tight_layout()
    for i in range(cols):
        for j, row in data.iterrows():
            x_selected = X_train[y_train==j]
            axs[j][i].imshow(x_selected[np.random.randint(0, len(x_selected)-1), :, :], cmap = plt.get_cmap('gray'))
            axs[j][i].axis("off")
            if i == 2:
                num_samples.append(len(x_selected))
    plt.show()
    return num_samples
    
    
def check_data(train_data, val_data, test_data):
    X_train, y_train = train_data['features'], train_data['labels']
    X_val, y_val = val_data['features'], val_data['labels']
    X_test, y_test = test_data['features'], test_data['labels']
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    assert(X_train.shape[0] == y_train.shape[0]), "The number of training images does not match the number of labels"
    assert(X_val.shape[0] == y_val.shape[0]), "The number of validation images does not match the number of labels"
    assert(X_test.shape[0] == y_test.shape[0]), "The number of test images does not match the number of labels"
    assert(X_train.shape[1:] == (32, 32, 3)), "The dimensions of the training images are not 32 x 32 x 3"
    assert(X_val.shape[1:] == (32, 32, 3)), "The dimensions of the validation images are not 32 x 32 x 3"
    assert(X_test.shape[1:] == (32, 32, 3)), "The dimensions of the test images are not 32 x 32 x 3"
    return X_train, y_train, X_val, y_val, X_test, y_test
    
    
def unpickle():
    with open('german_traffic_signs/train.p', 'rb') as f:
        train_data = pickle.load(f)
    with open('german_traffic_signs/valid.p', 'rb') as f:
        val_data = pickle.load(f)
    with open('german_traffic_signs/test.p', 'rb') as f:
        test_data = pickle.load(f)
    return train_data, val_data, test_data


if __name__ == "__main__":
    main()