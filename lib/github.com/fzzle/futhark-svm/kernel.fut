module type kernel = {
  -- | Kernel parameters.
  type s
  -- | Float type.
  type t
  -- | Compute a single kernel value.
  val value [n]: s -> [n]t -> [n]t -> t
  -- | Compute the kernel diagonal.
  val diag [n][m]: s -> [n][m]t -> *[n]t
  -- | Extract diagonal from a full kernel matrix.
  val extdiag [n]: s -> [n][n]t -> *[n]t
  -- | Compute a row of the kernel matrix.
  val row [n][m]: s -> [n][m]t -> [m]t -> [n]t -> t -> *[n]t
  -- | Compute the kernel matrix.
  val matrix [n][m][o]: s->[n][m]t->[o][m]t->[n]t->[o]t-> *[n][o]t
}

-- | Defines the commonly used methods for kernels.
module default (F: float) = {
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

-- | The linear kernel: K(u, v) = <u, v>
module linear (F: float): kernel = {
  module T = default F

  type t = F.t
  type s = ()

  let value   _     = T.dot
  let extdiag _     = T.extdiag
  let diag   (p: s) = T.diag   (value p)
  let row    (p: s) = T.row    (value p)
  let matrix (p: s) = T.matrix (value p)
}

-- | The rbf kernel with simple squared distance calculated by:
--   K(u, v) = ||u - v||² = \sum (u_i - v_i) * (u_i - v_i).
module rbf_simple (F: float): kernel = {
  module T = default F

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

  let row    (p: s) = T.row    (value p)
  let matrix (p: s) = T.matrix (value p)
}

-- | The rbf kernel with squared distance calculated by:
--   K(u, v) = ||u - v||² = <u, u> + <v, v> - 2 * <u, v>
module rbf (F: float): kernel = {
  module T = default F

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

  let row [n][m] ({gamma}: s) (X0: [n][m]t)
      (u: [m]t) (D0: [n]t) (d_u: t): *[n]t =
    let k x = F.exp (F.negate gamma F.* x)
    in map2 (\v d_v -> k F.(d_u + d_v - i32 2 * T.dot u v)) X0 D0

  let matrix [n][m][o] (p: s) (X0: [n][m]t)
      (X1: [o][m]t) (D0: [n]t) (D1: [o]t): *[n][o]t =
    replicate n (replicate o (F.i32 0))
}
