from futhark_data import dump
from futhark_ffi import Futhark
import numpy as np
import sys
import _fsvm

fsvm = Futhark(_fsvm)

kernels = {
  'linear': 0,
  'sigmoid': 1,
  'polynomial': 2,
  'rbf': 3
}

class SVC():
  # TODO: Inherit from sklearn.BaseEstimator
  def __init__(self, kernel='rbf', C=10., gamma=0.1,
      coef0=0., degree=3., eps=0.001, max_iter=100000000,
      verbose=False, n_ws=1024):
    """
    ke
    """
    if kernel not in kernels:
      raise Exception()

    self.kernel = kernel
    self.fit_ = getattr(fsvm, f'svc_{kernel}_fit')
    self.predict_ = getattr(fsvm, f'svc_{kernel}_predict')

    self.C = C
    self.n_ws = n_ws
    self.eps = eps
    self.max_t = max_iter
    self.max_t_out = 10000
    self.max_t_in = 100000
    self.verbose = verbose
    self.gamma = gamma
    self.coef0 = coef0
    self.degree = degree

    self.trained = False

  def fit_input_dump(self, X, y, name):
    f = open(f'data/{name}', 'wb')
    dump(X.astype(np.float32), f, binary=True)
    dump(y.astype(np.int32), f, binary=True)
    # dump(np.dtype('float32').type(self.C), f, binary=True)
    # dump(np.dtype('int32').type(self.n_ws), f, binary=True)
    # dump(np.dtype('int32').type(self.max_t), f, binary=True)
    # dump(np.dtype('int32').type(self.max_t_in), f, binary=True)
    # dump(np.dtype('int32').type(self.max_t_out), f, binary=True)
    # dump(np.dtype('float32').type(self.eps), f, binary=True)
    # dump(np.dtype('float32').type(self.gamma), f, binary=True)
    # dump(np.dtype('float32').type(self.coef0), f, binary=True)
    # dump(np.dtype('float32').type(self.degree), f, binary=True)
    f.close()

  def predict_input_dump(self, X, name):
    # f = open(f'./data/{name}', 'wb')
    # dump(X.astype(np.float32), f, binary=True)
    # dump(fsvm.from_futhark(self.__S), f, binary=True)
    # dump(fsvm.from_futhark(self.__A), f, binary=True)
    # dump(fsvm.from_futhark(self.__I), f, binary=True)
    # dump(fsvm.from_futhark(self.__rhos), f, binary=True)
    # dump(fsvm.from_futhark(self.__sizes), f, binary=True)
    # dump(np.dtype('int32').type(self.__n_classes), f, binary=True)
    # dump(np.dtype('int32').type(self.kernel), f, binary=True)
    # dump(np.dtype('float32').type(self.gamma), f, binary=True)
    # dump(np.dtype('float32').type(self.coef0), f, binary=True)
    # dump(np.dtype('float32').type(self.degree), f, binary=True)
    # dump(np.dtype('int32').type(ws), f, binary=True)
    # f.close()
    pass

  def fit(self, X, y):

    res = self.fit_(
      X, y,
      self.C,
      self.n_ws,
      self.max_t,
      self.max_t_in,
      self.max_t_out,
      self.eps,
      self.gamma,
      self.coef0,
      self.degree
    )

    A, I, S, Z, R, n_c, obj, T, t_out = res
    if self.verbose:
      _rhos = fsvm.from_futhark(R)
      obj  = fsvm.from_futhark(obj)
      iter = fsvm.from_futhark(T)
      t_out = fsvm.from_futhark(t_out)
      n_sv = len(fsvm.from_futhark(S))
      sizes = fsvm.from_futhark(Z)
      print('objective values:\n', obj)
      print('rhos:\n', _rhos)
      print('inner iterations:\n', iter)
      print('outer iterations:\n', t_out)
      print('sv:\n', sizes)
      print('nSV:', n_sv)
      print('avg. obj.:', np.mean(obj))
      print('total iterations:', np.sum(iter))
      print('total outer iter:', np.sum(t_out))
    self.__S = S # support_vectors_
    self.__I = I # support_
    self.__A = A # coef_
    self.__Z = Z
    self.__R = R # intercept_
    self.__n_c = n_c
    self.trained = True

    if profiling:
      f = open('profile.txt', 'w+')
      errptr = fsvm.lib.futhark_context_report(fsvm.ctx)
      errstr = fsvm.ffi.string(errptr).decode()
      fsvm.lib.free(errptr)
      f.write(errstr)
      f.close()


  def predict(self, X, n_ws=64):



    if not self.trained:
      raise Exception('Not trained')
    p = self.predict_(X,
      self.__A,
      self.__I,
      self.__S,
      self.__Z,
      self.__R,
      self.__n_c,
      n_ws,
      self.gamma,
      self.coef0,
      self.degree
    )
    return fsvm.from_futhark(p)

  # def fit_input_dump(self):

  # def predict_input_dump(self):

  @property
  def support_vectors_(self):
    # TODO: Find exact format in sklearn
    return fsvm.from_futhark(self.__S)
