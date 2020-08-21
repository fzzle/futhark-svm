from futhark_data import dump
from futhark_ffi import Futhark
import numpy as np
import _fsvm

fsvm = Futhark(_fsvm)

kernels = {
  'linear': 0,
  'sigmoid': 1,
  'polynomial': 2,
  'rbf': 3
}

class SVC():

  # Negative values doesn't work for primitives w/ futhark_ffi
  # it seems, so max_t_out is set to a high value instead of -1.
  def __init__(self, kernel='rbf', C=10., n_ws=1024,
      max_t=100000000, max_t_in=102400, max_t_out=100000000,
      eps=0.001, gamma='auto', coef0=0., degree=3., verbose=0.):
    if kernel not in kernels:
      raise Exception('Unknown kernel')
    self.kernel = kernel
    self.__fit     = getattr(fsvm, f'svc_{kernel}_fit')
    self.__predict = getattr(fsvm, f'svc_{kernel}_predict')

    self.C = C

    self.n_ws = n_ws
    self.max_t = max_t
    self.max_t_in = max_t_in
    self.max_t_out = max_t_out
    self.eps = eps

    self.gamma = gamma
    self.coef0 = coef0
    self.degree = degree

    self.verbose = verbose
    self.trained = False

  def fit(self, X, y):
    if self.gamma == 'auto':
      self.__gamma = 1.0 / X.shape[1]
    else:
      self.__gamma = self.gamma

    output = self.__fit(
      X, y,
      self.C,
      self.n_ws,
      self.max_t,
      self.max_t_in,
      self.max_t_out,
      self.eps,
      self.__gamma,
      self.coef0,
      self.degree
    )

    # Futhark output in C format.
    A, I, S, Z, R, n_c, O, T, T_out = output

    self.__S = S # support_vectors_
    self.__I = I # support_
    self.__A = A # coef_
    self.__Z = Z
    self.__R = R # intercept_
    self.__n_c = n_c
    self.trained = True

    if self.verbose <= 0: return

    _S = fsvm.from_futhark(S)
    _O = fsvm.from_futhark(O)
    _T = fsvm.from_futhark(T)
    _T_out = fsvm.from_futhark(T_out)

    print('total support vecs: ', len(_S))
    print('avg. objective vals:', np.mean(_O))
    print('total inner iter:   ', np.sum(_T))
    print('total outer iter:   ', np.sum(_T_out))

    if self.verbose <= 1: return

    _R = fsvm.from_futhark(R)
    _Z = fsvm.from_futhark(Z)

    print('intercepts:\n       ', _R)
    print('objectives values:\n', _O)
    print('# support vectors:\n', _Z)
    print('inner iterations:\n ', _T)
    print('outer iterations:\n ', _T_out)

  def dump_fit(self, X, y, fn):
    if self.gamma == 'auto':
      self.__gamma = 1.0 / X.shape[1]
    else:
      self.__gamma = self.gamma

    f = open(fn, 'wb')
    dump(X.astype(np.float32), f, binary=True)
    dump(y.astype(np.int32), f, binary=True)
    dump(np.dtype('float32').type(self.C), f, binary=True)
    dump(np.dtype('int32').type(self.n_ws), f, binary=True)
    dump(np.dtype('int32').type(self.max_t), f, binary=True)
    dump(np.dtype('int32').type(self.max_t_in), f, binary=True)
    dump(np.dtype('int32').type(self.max_t_out), f, binary=True)
    dump(np.dtype('float32').type(self.eps), f, binary=True)
    dump(np.dtype('float32').type(self.__gamma), f, binary=True)
    dump(np.dtype('float32').type(self.coef0), f, binary=True)
    dump(np.dtype('float32').type(self.degree), f, binary=True)
    f.close()

  def predict(self, X, n_ws=64):
    if not self.trained:
      raise Exception('Not trained')

    output = self.__predict(
      X,
      self.__A,
      self.__I,
      self.__S,
      self.__Z,
      self.__R,
      self.__n_c,
      n_ws,
      self.__gamma,
      self.coef0,
      self.degree
    )

    return fsvm.from_futhark(output)

  def dump_predict(self, X, fn, n_ws=64):
    if not self.trained:
      raise Exception('Not trained')

    f = open(fn, 'wb')
    dump(X.astype(np.float32), f, binary=True)
    dump(fsvm.from_futhark(self.__A), f, binary=True)
    dump(fsvm.from_futhark(self.__I), f, binary=True)
    dump(fsvm.from_futhark(self.__S), f, binary=True)
    dump(fsvm.from_futhark(self.__Z), f, binary=True)
    dump(fsvm.from_futhark(self.__R), f, binary=True)
    dump(np.dtype('int32').type(self.__n_c), f, binary=True)
    dump(np.dtype('int32').type(n_ws), f, binary=True)
    dump(np.dtype('float32').type(self.gamma), f, binary=True)
    dump(np.dtype('float32').type(self.coef0), f, binary=True)
    dump(np.dtype('float32').type(self.degree), f, binary=True)
    f.close()
