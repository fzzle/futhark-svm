import "../../lib/github.com/fzzle/futhark-svm/svm"

module fsvm = svm f32
module K = fsvm.kernels
module L = fsvm.svc K.linear

-- Matches libsvm?
-- ==
-- entry: fit_iris_linear
-- compiled input @ iris.data
-- output { [-0.748057f32, -0.203684f32, -15.759854f32] }
entry fit_iris_linear X y =
  let {weights=_, details} = L.fit X y 1 fsvm.default_fit {}
  in details.O

module P = fsvm.svc K.polynomial

-- libsvm: -26.0443f32, got -24.302277f32
-- ==
-- entry: fit_iris_polynomial
-- compiled input @ iris.data
-- output { [-0.000147, -0.000028, -24.302277f32] }
entry fit_iris_polynomial X y =
  let {weights=_, details} = P.fit X y 10 fsvm.default_fit
    {gamma=1, coef0=0, degree=3}
  in details.O

module R = fsvm.svc K.rbf

-- ==
-- entry: fit_iris_rbf
-- compiled input @ iris.data
-- output { [-3.143738f32, -3.592592f32, -74.489527f32] }
entry fit_iris_rbf X y =
  let {weights=_, details} = R.fit X y 10 fsvm.default_fit {gamma=1}
  in details.O