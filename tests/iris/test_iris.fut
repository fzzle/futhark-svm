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