import "../lib/github.com/fzzle/futhark-svm/svm"

module fsvm = svm f32
module K = fsvm.kernels
module L = fsvm.svc K.linear

-- Simple sanity test.
-- ==
-- entry: fit_predict_linear
-- input { [[0f32, 1f32], [1f32, 0f32]]
--          [0i64, 1i64]
--         [[0f32, 1f32], [1f32, 0f32]] }
-- output { [0i64, 1i64] }
entry fit_predict_linear [m] (X: [][m]f32) y (X_test: [][m]f32) =
  let {weights, details=_} = L.fit X y 1 fsvm.default_fit {}
  in L.predict X_test weights fsvm.default_predict {}
