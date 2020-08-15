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

module P = fsvm.svc K.rbf

-- Objective values won't be entirely the same, but it's close..
-- libsvm: [-3.143738f32, -3.592592f32, -74.489527f32]
-- fsvm:   [-3.155767f32, -3.596976f32, -74.02107f32]
-- ==
-- entry: fit_iris_rbf
-- compiled input @ iris.data
-- output { [-3.155767f32, -3.596976f32, -74.02107f32] }
entry fit_iris_rbf X y =
  let {weights=_, details} = P.fit X y 10 fsvm.default_fit {gamma=1}
  in details.O