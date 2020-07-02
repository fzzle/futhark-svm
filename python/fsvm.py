from futhark_ffi import Futhark
from pathlib import Path
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'bin'))

import _main

fsvm = Futhark(_main)

#fsvm.lib.futhark_context_config_add_build_option(fsvm.conf, b'-cl-fast-relaxed-math')

kernels = {
  'rbf': 0,
  'linear': 1,
  'polynomial': 2,
  'sigmoid': 3
}

class SVC():
  def __init__(self, kernel='rbf', C=10, gamma=0.1,
      coef0=0, degree=3, eps=0.001, max_iter=100000000,
      verbose=False):
    if kernel not in kernels:
      raise Exception()
    self.kernel = kernels.get(kernel)
    self.C = C
    self.gamma = gamma
    self.coef0 = coef0
    self.degree = degree
    self.eps = eps
    self.max_iter = max_iter
    self.verbose = verbose
    self.trained = False

  def fit(self, X, y):
    if self.trained:
      raise Exception('Already trained')
    X = X.astype(np.float32)
    y = y.astype(np.uint8)
    res = fsvm.train(X, y, self.kernel, self.C,
      self.gamma, self.coef0, self.degree,
      self.eps, self.max_iter)
    (A, I, S, flags, rhos, obj, iter, t) = res
    if self.verbose:
      _rhos = fsvm.from_futhark(rhos)
      obj  = fsvm.from_futhark(obj)
      iter = fsvm.from_futhark(iter)
      print('total iterations:', np.sum(iter))
      print('rhos:\n', _rhos)
      print('objective values:\n', obj)
      print('iterations:\n', iter)
    self.__support_vectors = S
    self.__sv_indices = I
    self.__alphas = A
    self.__flags = flags
    self.__rhos = rhos
    self.__n_classes = t
    self.trained = True

  def predict(self, X, ws=64):
    if not self.trained:
      raise Exception('Not trained')
    p = fsvm.predict(X,
      self.__support_vectors,
      self.__alphas,
      self.__sv_indices,
      self.__rhos,
      self.__flags,
      self.__n_classes,
      self.kernel,
      self.gamma,
      self.coef0,
      self.degree,
      ws
    )
    return fsvm.from_futhark(p)
