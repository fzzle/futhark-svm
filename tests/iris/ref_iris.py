from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()

# Reference for test #1 in test_iris.fut.
c0 = SVC(kernel='linear', C=1, verbose=True)
c0.fit(iris.data, iris.target)

# Polynomial ref.
c1 = SVC(kernel='poly', C=10, gamma=0.1, degree=3, verbose=True)
c1.fit(iris.data, iris.target)

# RBF ref.
c1 = SVC(kernel='rbf', C=10, gamma=1, verbose=True)
c1.fit(iris.data, iris.target)