module type kernel_function = {
  -- | Kernel parameters.
  type s
  -- | Float type.
  type t
  -- | Compute a single kernel value.
  val value [n]: s -> [n]t -> [n]t -> t
}

module type kernel = {
  include kernel_function
  -- | Compute the kernel diagonal.
  val diag [n][m]: s -> [n][m]t -> *[n]t
  -- | Extract diagonal from a full kernel matrix.
  val extdiag [n]: s -> [n][n]t -> *[n]t
  -- | Compute a row of the kernel matrix.
  val row [n][m]: s -> [n][m]t -> [m]t -> [n]t -> t -> *[n]t
  -- | Compute the kernel matrix.
  val matrix [n][m][o]: s ->[n][m]t->[o][m]t->[n]t->[o]t-> *[n][o]t
}

-- | Kernel utilities.
module kernel_util (F: float) = {
  type t = F.t
  -- | Value function type.
  type^ f [n] = [n]t -> [n]t -> t
  -- | Squared euclidean distance.
  let sqdist [n] (u: [n]t) (v: [n]t): t =
    F.(sum (map (\x -> x * x) (map2 (-) u v)))

  -- | Vector dot product.
  let dot [n] (u: [n]t) (v: [n]t): t =
    F.sum (map2 (F.*) u v)

  -- | Extract the diagonal of a symmetric kernel matrix.
  let extdiag [n] (K: [n][n]t): *[n]t =
    map (\i -> K[i, i]) (iota n)

  -- | Compute diagonal of a kernel matrix for samples in X.
  let diag [n][m] (k: f [m]) (X: [n][m]t): *[n]t =
    map (\u -> k u u) X

  -- | Compute a full kernel matrix row.
  let row [n][m] (k: f [m]) (X: [n][m]t)
      (v: [m]t) (_: [n]t) (_: t): *[n]t =
    map (\u -> k u v) X

  let matrix [n][m][o] (k: f [m]) (X0: [n][m]t)
      (X1: [o][m]t) (_: [n]t) (_: [o]t): *[n][o]t =
    map (\u -> map (k u) X1) X0
}

-- | Parametric module which makes it possible to implement a new
-- kernel type only by providing the kernel function and the model
-- parameters of the function.
module default (F: float) (K: kernel_function with t = F.t) = {
  module T = kernel_util F

  type t = K.t
  type s = K.s

  let value = K.value

  let extdiag _     = T.extdiag
  let diag   (p: s) = T.diag   (value p)
  let row    (p: s) = T.row    (value p)
  let matrix (p: s) = T.matrix (value p)
}

-- | The linear kernel:
-- K(u, v) = <u, v>
module linear (F: float): kernel = default F {
  module T = kernel_util F

  type t = F.t
  type s = ()

  let value _ = T.dot
}

-- | The polynomial kernel:
-- K(u, v) = (\gamma * <u, v> + c) ** degree
module polynomial (F: float): kernel = default F {
  module T = kernel_util F

  type t = F.t
  type s = {
    gamma: t,
    coef0: t,
    degree: t
  }

  let value [n] (p: s) (u: [n]t) (v: [n]t): t =
    F.((p.gamma * T.dot u v + p.coef0) ** p.degree)
}

-- | The sigmoid kernel:
-- K(u, v) = tanh (\gamma * <u, v> + c)
module sigmoid (F: float): kernel = default F {
  module T = kernel_util F

  type t = F.t
  type s = {
    gamma: t,
    coef0: t
  }

  let value [n] (p: s) (u: [n]t) (v: [n]t): t =
    F.(tanh (p.gamma * T.dot u v + p.coef0))
}

-- | The rbf kernel with simple squared distance calculated by:
-- K(u, v) = \exp(-\gamma * ||u - v||²)
--         = \exp(-\gamma * (\sum (u_i - v_i)²))
module rbf_simple (F: float): kernel = {
  module T = kernel_util F

  type t = F.t
  type s = {
    gamma: t
  }

  let value [n] ({gamma}: s) (u: [n]t) (v: [n]t): t =
    F.exp (F.negate gamma F.* T.sqdist u v)

  -- rbf diagonals are always all 1's.
  let extdiag [n] (_: s) (_: [n][n]t): *[n]t =
    replicate n (F.i32 1)

  let diag [n][m] (_: s) (_: [n][m]t): *[n]t =
    replicate n (F.i32 1)

  let row    (p: s) = T.row    (value p)
  let matrix (p: s) = T.matrix (value p)
}

-- | The rbf kernel where the matrix and row operations use
-- <u, u> + <v, v> - 2 * <u, v> to find the squared distance:
-- K(u, v) = \exp(-\gamma * (<u, u> + <v, v> - 2 * <u, v>))
module rbf (F: float): kernel = {
  module T = kernel_util F

  type t = F.t
  type s = {
    gamma: t
  }

  let value [n] ({gamma}: s) (u: [n]t) (v: [n]t): t =
    F.exp (F.negate gamma F.* T.sqdist u v)

  let extdiag [n] (_: s) (_: [n][n]t): *[n]t =
    replicate n (F.i32 1)

  let diag [n][m] (_: s) (_: [n][m]t): *[n]t =
    replicate n (F.i32 1)

  let row [n][m] ({gamma}: s) (X: [n][m]t)
      (u: [m]t) (D: [n]t) (d_u: t): *[n]t =
    let k x = F.exp (F.negate gamma F.* x)
    in map2 (\v d_v -> k F.(d_u + d_v - i32 2 * T.dot u v)) X D

  let matrix [n][m][o] (p: s) (X0: [n][m]t)
      (X1: [o][m]t) (D0: [n]t) (D1: [o]t): *[n][o]t =
    map2 (\u d_u -> row p X1 u D1 d_u) X0 D0
}
