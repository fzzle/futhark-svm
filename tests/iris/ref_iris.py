from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()

# Reference for test #1 in test_iris.fut.
c = SVC(kernel='linear', C=1, verbose=True)
c.fit(iris.data, iris.target)