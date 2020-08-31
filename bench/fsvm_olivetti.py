from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from futhark_svm import SVC
import numpy as np
import time
import json

dt = fetch_olivetti_faces()
X = dt.data
y = dt.target

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.1, random_state=0)

with open('settings/olivetti.json') as f:
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

  t = m.predict(X_train)
  print('training error:     ', 1-accuracy_score(y_train, t))

  start = time.time()
  p = m.predict(X_test)
  end = time.time()
  print('prediction time:    ', end - start)
  print('accuracy:           ', accuracy_score(y_test, p))
