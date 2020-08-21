from sklearn.metrics import accuracy_score
from idx2numpy import convert_from_file
from futhark_svm import SVC
import numpy as np
import time
import json

X_train = convert_from_file('./data/train-images-idx3-ubyte')
y_train = convert_from_file('./data/train-labels-idx1-ubyte')
X_train = X_train.reshape(60000, 784)

X_test = convert_from_file('./data/t10k-images-idx3-ubyte')
y_test = convert_from_file('./data/t10k-labels-idx1-ubyte')
X_test = X_test.reshape(10000, 784)

with open('settings.json') as f:
  data = json.load(f)

for s in data['models']:
  print(s)

  m = SVC(
    kernel = s.get('kernel', 'rbf'),
    C      = s.get('C', 10.0),
    coef0  = s.get('coef0', 0.0),
    gamma  = s.get('gamma', 'auto'),
    degree = float(s.get('degree', 3)),
    verbose= True
  )

  start = time.time()
  m.fit(X_train, y_train)
  end = time.time()
  print('fit time:           ', end - start)

  start = time.time()
  p = m.predict(X_test)
  end = time.time()
  print('prediction time:    ', end - start)
  print('accuracy:           ', accuracy_score(y_test, p))
