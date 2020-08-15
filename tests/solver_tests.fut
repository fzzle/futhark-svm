import "../lib/github.com/fzzle/futhark-svm/svm"
import "../lib/github.com/fzzle/futhark-svm/solver"
import "../lib/github.com/fzzle/futhark-svm/kernel"

module fsvm = svm f32
module L = solver f32 fsvm.kernels.linear

-- ==
-- entry: solve_linear
-- input { [[0f32, 1f32], [1f32, 0f32],
--          [-1f32, 0f32], [0f32, -1f32]]
--         [1f32, 1f32, -1f32, -1f32] }
-- output { [1f32, 0f32, -1f32, 0f32] }
entry solve_linear X y =
  L.solve X y (1, 1) fsvm.default_training {} |> (.0)
