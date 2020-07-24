import numpy as np
import sys
import os
from fsvm import SVC

train = np.loadtxt('./data/mnist_train.csv', delimiter=',')

y_train = train[:, 0]
X_train = train[:, 1:] * (0.99 / 255) + 0.01

m = SVC(kernel='linear', degree=3, C=10, verbose=True)

m.fit(X_train, y_train)
