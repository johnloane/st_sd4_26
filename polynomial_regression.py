import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(0)

def main():
    X, y = generate_and_plot_data()
    build_and_test_model(X, y)
    
    
def build_and_test_model(X, y):
    model = Sequential()
    model.add(Dense(50, input_dim=1, activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(1))
    adam = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer = adam)
    model.fit(X, y, epochs=50)
    predictions = model.predict(X)
    plt.scatter(X, y)
    plt.plot(X, predictions, 'ro')
    plt.show()
    

def generate_and_plot_data():
    points = 500
    X = np.linspace(-3, 3, points)
    y = np.sin(X) + np.random.uniform(-0.5, 0.5, points)
    plt.scatter(X, y)
    plt.show()
    return X, y

if __name__ == "__main__":
    main()