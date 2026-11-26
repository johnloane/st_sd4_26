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
import matplotlib.image as mpimg

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }
from imgaug import augmenters as iaa

num_bins = 25
samples_per_bin = 500
datadir = "C:\\Users\\johnl\\dev\\Smart Technologies\\st_26\\SimulatorTrainingData\\"


def main():
    data = load_data()
    bins, centre = bin_and_plot_data(data)
    balanced_data = balance_data(data, bins)
    #plot_balanced_data(balanced_data, centre)
    X_train, X_valid, y_train, y_valid, image_paths = split_data(balanced_data)
    #plot_validation_training_distribution(y_train, y_valid)
    #show_original_and_preprocessed_sample_image(image_paths)
    #X_train, X_valid = apply_preprocessing(X_train, X_valid)
    model = nvidia_model()
    train_and_test_model(model, X_train, y_train, X_valid, y_valid)
    display_augmented_and_original(image_paths, y_train)
    
    
def display_augmented_and_original(image_paths, steering_angle):
    random_index = random.randint(0, 3000)
    image = image_paths[random_index]
    steering_angle = steering_angle[random_index]
    original_image = mpimg.imread(image)
    augmented_image, steering_angle = random_augment(image, steering_angle)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image" + str(steering_angle))
    axs[1].imshow(augmented_image)
    axs[1].set_title("Augmented Image" + str(steering_angle))
    plt.show()
    
     
    
def train_and_test_model(model, X_train, y_train, X_valid, y_valid):
    print(model.summary)
    history = model.fit(batch_generator(X_train, y_train, 100, 1), steps_per_epoch=100, epochs=20, validation_data=batch_generator(X_valid, y_valid, 100, 0), validation_steps = 100, verbose=1, shuffle=1)
    model.save('basic_augmented_nvidia_model.keras')
    # This is a regression problem so accuracy is not an appropriate metric
    # You would have to get the steering angle exactly right for it to be
    # counted as accurate. As there are an infinite number of steering angles
    # this is almost impossible. Plot the loss instead
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
    
def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5,5), strides=(2,2), input_shape=(66, 200, 3), activation="elu"))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation="elu"))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation="elu"))
    model.add(Conv2D(64, (3,3), activation="elu"))
    model.add(Conv2D(64, (3,3), activation="elu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    optimizer = Adam(learning_rate = 0.001)
    model.compile(loss='mse', optimizer = optimizer)
    return model
    
 
def apply_preprocessing(X_train, X_valid):
    X_train = np.array(list(map(img_preprocess, X_train)))
    X_valid = np.array(list(map(img_preprocess, X_valid)))   
    return X_train, X_valid
    
    

def show_original_and_preprocessed_sample_image(image_paths):
    image = image_paths[100]
    original_image = mpimg.imread(image)
    preprocessed_image = img_preprocess(image)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[1].imshow(preprocessed_image)
    axes[1].set_title("Preprocessed Image")
    plt.show()
    
    
def img_preprocess(img):
    img = mpimg.imread(img)
    img = img[60:135, :, :]
    # The NVIDIA paper recommends YUV rather than RBG
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

def img_preprocess_no_imread(img):
    img = img[60:135, :, :]
    # The NVIDIA paper recommends YUV rather than RBG
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
    
def  plot_validation_training_distribution(y_train, y_valid):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
    axes[0].set_title('Training data')
    axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
    axes[1].set_title('Validation data')
    plt.show()

    
def split_data(data):
    image_paths, steerings = load_steering_img(datadir + 'IMG', data)
    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=77)
    print(f"Training samples {len(X_train)}, Validation samples {len(X_valid)}")
    return X_train, X_valid, y_train, y_valid, image_paths

def load_steering_img(datadir, data):
    image_path = []
    steerings = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data['center'], indexed_data['left'], indexed_data['right']
        image_path.append(os.path.join(datadir, center.strip()))
        image_path.append(os.path.join(datadir, left.strip()))
        image_path.append(os.path.join(datadir, right.strip()))
        steerings.append(float(indexed_data['steering']))
        steerings.append(float(indexed_data['steering']))
        steerings.append(float(indexed_data['steering']))
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
    data = data.drop(data.index[remove_list])
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


def zoom(image_to_zoom):
    zoom_func = iaa.Affine(scale=(1, 1.3))
    z_image = zoom_func.augment_image(image_to_zoom)
    return z_image

def pan(image_to_pan):
    pan_func = iaa.Affine(translate_percent={"x":(-0.1, 0.1), "y":(-0.1, 0.1)})
    pan_image = pan_func.augment_image(image_to_pan)
    return pan_image

def img_random_brightness(image_to_brighten):
    bright_func = iaa.Multiply((0.2, 1.2))
    bright_image = bright_func.augment_image(image_to_brighten)
    return bright_image

def img_random_flip(image_to_flip, steering_angle):
    #1 - flip horizontal, 0, vertical, -1 combo of both
    flipped_image = cv2.flip(image_to_flip, 1)
    steering_angle = - steering_angle
    return flipped_image, steering_angle

def random_augment(image_to_augment, steering_angle):
    augment_image = mpimg.imread(image_to_augment)
    if np.random.rand() < 0.5:
        augment_image = zoom(augment_image)
    if np.random.rand() < 0.5:
        augment_image = pan(augment_image)
    if np.random.rand() < 0.5:
        augment_image = img_random_brightness(augment_image)
    if np.random.rand() < 0.5:
        augment_image, steering_angle = img_random_flip(augment_image, steering_angle)  
    return augment_image, steering_angle  


def batch_generator(image_paths, steering_angles, batch_size, is_training):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths)-1)
            if is_training:
                im, steering = random_augment(image_paths[random_index], steering_angles[random_index])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_angles[random_index]
            
            img = img_preprocess_no_imread(im)  
            batch_img.append(img)
            batch_steering.append(steering)
        yield(np.asarray(batch_img), np.asarray(batch_steering))  
    

if __name__ == "__main__":
    main()