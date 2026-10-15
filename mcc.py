import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:, 0])+ 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:, 1])+ 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = np.argmax(model.predict(grid), axis=-1)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
    plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
    plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
    x = -1
    y = 1
    point = np.array([[x, y]])
    prediction = model.predict(point)
    plt.plot([x], [y], marker="o", markersize=10, color="red")
    print(f"Prediction is {prediction}")
    plt.show()
    
    
np.random.seed(0)
n_pts = 500

centres = [[-1, 1], [-1, -1], [1, -1], [1, 1], [0, 0]]
points, labels = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centres, cluster_std = 0.4)
plt.scatter(points[labels==0, 0], points[labels==0, 1])
plt.scatter(points[labels==1, 0], points[labels==1, 1])
plt.scatter(points[labels==2, 0], points[labels==2, 1])
plt.scatter(points[labels==3, 0], points[labels==3, 1])
plt.scatter(points[labels==4, 0], points[labels==4, 1])
plt.show()

cat_labels = to_categorical(labels, 5)
print(cat_labels) 

model = Sequential()
model.add(Dense(units=5, input_shape=(2,), activation='softmax'))
model.compile(Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
h = model.fit(x=points, y=cat_labels, verbose=1, batch_size=50, epochs=90)
plt.plot(h.history['accuracy'])
plt.xlabel('epoch')
plt.legend('Accuracy')
plt.title('Accuracy')
plt.show()

plt.plot(h.history['loss'])
plt.xlabel('epoch')
plt.legend('Loss')
plt.title('Loss')
plt.show()

plot_decision_boundary(points, cat_labels, model)
