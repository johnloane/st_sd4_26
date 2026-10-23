import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:, 0])+ 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:, 1])+ 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
    plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
    plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
    x = 0
    y = 0
    point = np.array([[x, y]])
    prediction = model.predict(point)
    plt.plot([x], [y], marker="o", markersize=10, color="red")
    plt.show()
    print(f"Prediction is {prediction}")

np.random.seed(0)
n_pts = 500
points, labels = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.3, factor=0.2)
print(points)
print(labels)
plt.scatter(points[labels==0, 0], points[labels==0, 1])
plt.scatter(points[labels==1, 0], points[labels==1, 1])
plt.show()

model = Sequential()
model.add(Dense(4, input_shape=(2, ), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(Adam(learning_rate=0.01), 'binary_crossentropy', metrics=['accuracy'])
h = model.fit(x=points, y = labels, verbose=1, batch_size=20, epochs=150, shuffle='true')
plt.plot(h.history['accuracy'])
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.title('Accuracy')
plt.show()

plot_decision_boundary(points, labels, model)