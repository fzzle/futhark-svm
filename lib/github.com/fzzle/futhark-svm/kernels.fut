type kernel =
    #rbf
  | #linear
  | #polynomial
  | #sigmoid

let kernel_from_id (id: i32): kernel =
  match (assert (id >= 0 && id < 4) id)
    case 0 -> #rbf
    case 1 -> #linear
    case 2 -> #polynomial
    case _ -> #sigmoid

-- Vector dot product.
local let dot [n] (a: [n]f32) (b: [n]f32): f32 =
  f32.sum (map2 (*) a b)

-- Squared euclidean distance.
local let sqdist [n] (a: [n]f32) (b: [n]f32): f32 =
  f32.sum (map (\x -> x * x) (map2 (-) a b))

-- Compute the linear kernel matrix.
local let linear [n][m][o] (X0: [n][m]f32)
    (X1: [o][m]f32): [n][o]f32 =
  map (\x -> map (dot x) X1) X0

local let rbf [n][m][o] (X0: [n][m]f32) (X1: [o][m]f32)
    (gamma: f32): [n][o]f32 =
  let f a b = f32.exp (-gamma * sqdist a b)
  in map (\x -> map (f x) X1) X0

local let polynomial [n][m][o] (X0: [n][m]f32) (X1: [o][m]f32)
    (gamma: f32) (coef0: f32) (degree: f32): [n][o]f32 =
  let f a b = (gamma * dot a b + coef0) ** degree
  in map (\x -> map (f x) X1) X0

local let sigmoid [n][m][o] (X0: [n][m]f32) (X1: [o][m]f32)
    (gamma: f32) (coef0: f32): [n][o]f32 =
  let f a b = f32.tanh (gamma * dot a b + coef0)
  in map (\x -> map (f x) X1) X0

-- Get kernel matrix for a dataset. K[i, j] corresponds to the
-- kernel value computed between samples x_i and x_j.
let kernel_matrix [n][m][o] (X0: [n][m]f32) (X1: [o][m]f32)
    (k: kernel) (gamma: f32) (coef0: f32)
    (degree: f32): [n][o]f32 =
  match k
    case #rbf        -> rbf X0 X1 gamma
    case #linear     -> linear X0 X1
    case #polynomial -> polynomial X0 X1 gamma coef0 degree
    case #sigmoid    -> sigmoid X0 X1 gamma coef0
