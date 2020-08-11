import "../lib/github.com/fzzle/futhark-svm/kernel"

module L = linear f32

-- ==
-- entry: test_linear_value
-- input { [2f32] [2f32] } output { 4f32 }
-- input { [0f32, 1f32, 2f32] [0f32, 1f32, 2f32] } output { 5f32 }
entry test_linear_value [n] (u: [n]f32) (v: [n]f32): f32 =
  L.value () u v

-- ==
-- entry: test_linear_row
-- input { [[2f32, 1f32, 5f32],
--          [3f32, 4f32, 6f32]]
--          [0f32, 1f32, 10f32] }
-- output { [51f32, 64f32] }
entry test_linear_row [n][m] (X: [n][m]f32) (u: [m]f32): [n]f32 =
  L.row () X u (replicate n 0) 0

-- ==
-- entry: test_default_extdiag
-- input { [[0f32, 1f32, 2f32],
--          [3f32, 4f32, 5f32],
--          [6f32, 7f32, 8f32]] }
-- output { [0f32, 4f32, 8f32] }
entry test_default_extdiag [n] (K: [n][n]f32): [n]f32 =
  L.extdiag () K

module P = polynomial f32

-- ==
-- entry: test_polynomial_matrix
-- input { [[2f32, 3f32, 7f32, 9f32],
--          [4f32, 0f32, 2f32, -1f32]] }
-- output { [[204.49f32, 1.69f32],
--           [1.69f32, 4.41f32]] }
entry test_polynomial_matrix [n][m] (X: [n][m]f32): [n][n]f32 =
  let D_l = replicate n 0 -- unused
  in P.matrix {gamma=0.1, degree=2, coef0=0} X X D_l D_l

module R_pre = rbf_pre f32
module R = rbf f32

-- ==
-- entry: test_rbf_diag
-- input { [[1f32, 2f32, 10f32],
--          [2f32, 3f32, -1f32]] }
-- output { [1f32, 1f32] }
entry test_rbf_diag [n][m] (X: [n][m]f32): [n]f32 =
  R.diag {gamma=0.1} X

-- ==
-- entry: test_rbf_pre_matrix test_rbf_matrix
-- input { [[2f32, -3f32, 4f32],
--          [3f32, 4f32, 5.5f32],
--          [1f32, 2f32, 3f32]] }
-- output { [[1f32, 0.0053803604f32, 0.06720551f32],
--           [0.0053803604f32, 1f32, 0.24050845f32],
--           [0.06720551f32, 0.24050845f32, 1f32]] }
entry test_rbf_pre_matrix [n][m] (X: [n][m]f32): [n][n]f32 =
  let D_l = L.diag () X
  in R_pre.matrix {gamma=0.1} X X D_l D_l

entry test_rbf_matrix [n][m] (X: [n][m]f32): [n][n]f32 =
  let D_l = replicate n 0
  in R.matrix {gamma=0.1} X X D_l D_l
