import "../lib/github.com/fzzle/futhark-svm/svm"
import "../lib/github.com/fzzle/futhark-svm/solver"
import "../lib/github.com/fzzle/futhark-svm/kernel"

module fsvm = svm f32
module K = fsvm.kernels
module L = solver f32 K.linear

-- Simple examples from "SVM Example" by Dan Ventura, 2009.
-- ==
-- entry: solve_linear
-- input { [[0f32, 1f32], [1f32, 0f32],
--          [-1f32, 0f32], [0f32, -1f32]]
--          [1f32, 1f32, -1f32, -1f32] }
-- output { [1f32, 0f32, -1f32, 0f32] 0f32 }
-- input { [[3f32, 1f32], [3f32, -1f32],
--          [6f32, 1f32], [6f32, -1f32],
--          [1f32, 0f32], [0f32, 1f32],
--          [0f32, -1f32], [-1f32, 0f32]]
--          [1f32, 1f32, 1f32, 1f32,
--           -1f32, -1f32, -1f32, -1f32] }
-- output { [0.25f32, 0.25f32, 0f32, 0f32,
--           -0.5f32, 0f32, 0f32, 0f32] 2f32 }
entry solve_linear X y =
  L.solve X y (1, 1) fsvm.default_training {}
  |> (\r -> (r.0, r.2)) -- A/b

module P = solver f32 K.polynomial

-- ==
-- entry: solve_polynomial
-- input { [[2f32, 2f32], [2f32, -2f32],
--          [-2f32, -2f32], [-2f32, 2f32],
--          [1f32, 1f32], [1f32, -1f32],
--          [-1f32, -1f32], [-1f32, 1f32]]
--          [1f32, 1f32, 1f32, 1f32,
--           -1f32, -1f32, -1f32, -1f32] 1f32 }
-- output { [1f32, 1f32, 1f32, 1f32,
--           -1f32, -1f32, -1f32, -1f32] 0f32 }
-- input { [[2f32, 2f32], [2f32, -2f32],
--          [-2f32, -2f32], [-2f32, 2f32],
--          [1f32, 1f32], [1f32, -1f32],
--          [-1f32, -1f32], [-1f32, 1f32]]
--          [1f32, 1f32, 1f32, 1f32,
--           -1f32, -1f32, -1f32, -1f32] 2f32 }
-- output { [0.0555f32, 0f32, 0f32, 0.0555f32,
--           0f32, 0f32, -0.0555f32, -0.0555f32] 1.6666f32 }
entry solve_polynomial X y degree =
  P.solve X y (1, 1) fsvm.default_training
    {gamma=1, coef0=0, degree}
  |> (\r -> (r.0, r.2))

