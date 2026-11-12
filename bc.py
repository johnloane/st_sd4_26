import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import cv2
import pandas as pd
import random
import os
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

num_bins = 25
samples_per_bin = 300
datadir = "C:\\Users\\johnl\\dev\\Smart Technologies\\st_26\\SimulatorTrainingData"


def main():
    data = load_data()
    bins, centre = bin_and_plot_data(data)
    balanced_data = balance_data(data, bins)
    plot_balanced_data(balanced_data, centre)
    X_train, X_valid, y_train, y_valid, image_paths = split_data(balanced_data)
    plot_validation_training_distribution(y_train, y_valid)
    
def  plot_validation_training_distribution(y_train, y_valid):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
    axes[0].title('Training data')
    axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
    axes[1].title('Validation data')
    plt.show()

    
def split_data(data):
    image_paths, steerings = load_steering_img(datadir, data)
    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=77)
    print(f"Training samples {len(X_train)}, Validation samples {len(X_valid)}")
    return X_train, X_valid, y_train, y_valid, image_paths

def load_steering_img(datadir, data):
    image_path = []
    steerings = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        print(indexed_data)
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steerings.append(float(indexed_data[3]))
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steerings)
    return image_paths, steerings     
   
def plot_balanced_data(balanced_data, centre):
    hist, _ = np.histogram(balanced_data['steering'], num_bins)
    plt.bar(centre, hist, width=0.05)
    plt.plot((np.min(balanced_data['steering']), np.max(balanced_data['steering'])), (samples_per_bin, samples_per_bin))
    plt.show()
    
    
def balance_data(data, bins):
    # Too many zeros, this would bias the model to pretty much always drive straight. This would be dangerous
    remove_list = []
    for i in range(num_bins):
        list_ = []
        for j in range(len(data['steering'])):
            if bins[i] <= data['steering'][j] <= bins[i+1]:
                list_.append(j)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)
    print("Remove: ", len(remove_list))
    data.drop(data.index[remove_list], inplace=True)
    print("Remaining: ", len(data))
    return data
        
        
    

def bin_and_plot_data(data):
    hist, bins = np.histogram(data['steering'], num_bins)
    print(bins)
    centre = (bins[:-1] + bins[1:])*0.5
    plt.bar(centre, hist, width=0.05)
    plt.show()
    return bins, centre
    
    
def load_data():
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
    pd.set_option('display.width', None)
    data['center'] = data['center'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)
    return data

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


if __name__ == "__main__":
    main()