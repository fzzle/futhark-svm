type kernel =
    #rbf
  | #linear
  | #polynomial
  | #sigmoid

-- Sum type?
type parameters =
  { kernel: kernel
  , gamma: f32
  , coef0: f32
  , degree: f32 }


-- module type _kernel = {

--   val prepare: ->

--   val f: i32 ->


-- }


-- | Vector dot product.
local let dot [n] (u: [n]f32) (v: [n]f32): f32 =
  f32.sum (map2 (*) u v)

-- | Squared euclidean distance.
local let sqdist [n] (u: [n]f32) (v: [n]f32): f32 =
  f32.sum (map (\x -> x * x) (map2 (-) u v))

local let rbf [n] (p: parameters) (u: [n]f32) (v: [n]f32): f32 =
  f32.exp (-p.gamma * sqdist u v)

local let polynomial [n] (p: parameters) (u: [n]f32) (v: [n]f32): f32 =
  (p.gamma * dot u v + p.coef0) ** p.degree

local let sigmoid [n] (p: parameters) (u: [n]f32) (v: [n]f32): f32 =
  f32.tanh (p.gamma * dot u v + p.coef0)

let kernel_from_id (id: i32): kernel =
  match (assert (id >= 0 && id < 4) id)
  case 0 -> #rbf
  case 1 -> #linear
  case 2 -> #polynomial
  case _ -> #sigmoid

let kernel_value [n] (p: parameters) (u: [n]f32) (v: [n]f32) =
  match p.kernel
  case #rbf        -> rbf p u v
  case #linear     -> dot u v
  case #polynomial -> polynomial p u v
  case #sigmoid    -> sigmoid p u v

let kernel_row [n][m] (p: parameters)
    (X: [n][m]f32) (u: [m]f32): [n]f32 =
  map (kernel_value p u) X

-- | Get kernel matrix for a dataset. K[i, j] corresponds to the
-- kernel value computed between samples x_i and x_j.
let kernel_matrix [n][m][o] (p: parameters)
    (X0: [n][m]f32) (X1: [o][m]f32): [n][o]f32 =
  map (kernel_row p X1) X0

-- | Get the diagonal of a full kernel matrix.
let kernel_diag [n] (p: parameters) (K: [n][n]f32): [n]f32 =
  match p.kernel
  case #rbf -> replicate n 1
  case _    -> map (\i -> K[i, i]) (iota n)

-- | Compute the diagonal of a kernel matrix.
let compute_kernel_diag [n][m] (p: parameters)
    (X: [n][m]f32): [n]f32 =
  match p.kernel
  case #rbf -> replicate n 1
  case _    -> map (\u -> kernel_value p u u) X
