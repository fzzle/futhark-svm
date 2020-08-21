from idx2numpy import convert_from_file
from futhark_svm import SVC

X_train = convert_from_file('./data/train-images-idx3-ubyte')
y_train = convert_from_file('./data/train-labels-idx1-ubyte')
X_train = X_train.reshape(60000, 784)

m = SVC(kernel='polynomial')
m.dump_fit(X_train, y_train, 'data/poly_mnist.data')
