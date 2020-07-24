from futhark_ffi import Futhark
from pathlib import Path
import numpy as np
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'bin'))

import _main

profiling = False
fsvm = Futhark(_main)

kernels = {
  'rbf': 0,
  'linear': 1,
  'polynomial': 2,
  'sigmoid': 3
}

class SVC():
  # TODO: Inherit from sklearn.BaseEstimator
  def __init__(self, kernel='rbf', C=10., gamma=0.1,
      coef0=0, degree=3, eps=0.001, max_iter=100000000,
      verbose=False):
    """
    ke
    """
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
    ""

    # res = fsvm.fit_gridsearch(X, y, self.kernel, np.array([self.C, 1, 100]),
    #   self.gamma, self.coef0, self.degree,
    #   self.eps, self.max_iter)


    res = fsvm.fit(X, y, self.kernel, self.C,
      self.gamma, self.coef0, self.degree,
      self.eps, self.max_iter)
    A, I, S, sizes, rhos, obj, iter, t = res
    if self.verbose:
      _rhos = fsvm.from_futhark(rhos)
      obj  = fsvm.from_futhark(obj)
      iter = fsvm.from_futhark(iter)
      n_sv = len(fsvm.from_futhark(S))
      print('nSV:', n_sv)
      print('avg. obj.:', np.mean(obj))
      print('total iterations:', np.sum(iter))
      print('objective values:\n', obj)
      print('rhos:\n', _rhos)
      print('iterations:\n', iter)
    self.__S = S # support_vectors_
    self.__I = I # support_
    self.__A = A # coef_
    self.__sizes = sizes
    self.__rhos = rhos         # intercept_
    self.__n_classes = t
    self.trained = True

    if profiling:
      f = open('profile.txt', 'w+')
      errptr = fsvm.lib.futhark_context_report(fsvm.ctx)
      errstr = fsvm.ffi.string(errptr).decode()
      fsvm.lib.free(errptr)
      f.write(errstr)
      f.close()

  def predict(self, X, ws=64):
    if not self.trained:
      raise Exception('Not trained')
    p = fsvm.predict(X,
      self.__S,
      self.__A,
      self.__I,
      self.__rhos,
      self.__sizes,
      self.__n_classes,
      self.kernel,
      self.gamma,
      self.coef0,
      self.degree,
      ws
    )
    return fsvm.from_futhark(p)

  @property
  def support_vectors_(self):
    # TODO: Find exact format in sklearn
    return fsvm.from_futhark(self.__S)
