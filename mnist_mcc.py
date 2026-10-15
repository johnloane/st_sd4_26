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
    #print_five_each_number(X_train, y_train)
    X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test)
    model = create_model()
    print(model.summary())
    train_and_evaluate(model, X_train, y_train, X_test, y_test)
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test score: ', score[0])
    print('Test accuracy: ', score[1])
    load_and_test_image(model)
    
    
def load_and_test_image(model):
    url = "https://colah.github.io/posts/2014-10-Visualizing-MNIST/img/mnist_pca/MNIST-p1815-3.png"
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    img_array = np.asarray(img)
    resized_image = cv2.resize(img_array, (28, 28))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    white_on_black = cv2.bitwise_not(grayscale_image)
    plt.imshow(white_on_black, cmap= plt.get_cmap('gray'))
    plt.show()
    white_on_black = white_on_black/255
    white_on_black = white_on_black.reshape(1, 784)
    prediction = np.argmax(model.predict(white_on_black), axis=1)
    print("Predicted digit: ", str(prediction))
    
    
    
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    h = model.fit(X_train, y_train, validation_split=0.1, epochs=40, batch_size=200, verbose=1, shuffle=1)
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title("Loss")
    plt.xlabel('epoch')
    plt.show()
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title("Accuracy")
    plt.xlabel('epoch')
    plt.show()
    
    
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
 
def preprocess(X_train, y_train, X_test, y_test):
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    X_train = X_train/255
    X_test = X_test/255
    X_train = X_train.reshape(X_train.shape[0], num_pixels)
    X_test = X_test.reshape(X_test.shape[0], num_pixels)
    return X_train, y_train, X_test, y_test
     
    
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