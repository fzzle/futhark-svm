
-- Kernel function type.
type^ f 't 's [n] = s -> [n]t -> [n]t -> t

module type parameters = {
  type s
}

module type kernel = {
  -- | Kernel parameters.
  type s
  -- | Float type.
  type t
  -- | Compute a single kernel value.
  val value [n]: f t s [n]
  -- | Compute the kernel diagonal.
  val diag [n][m]: s -> [n][m]t -> *[n]t
  -- | Extract diagonal from a full kernel matrix.
  val extdiag [n]: s -> [n][n]t -> *[n]t
  -- | Compute a row of the kernel matrix.
  val row [n][m]: s -> [n][m]t -> [m]t -> [n]t -> t -> *[n]t
  -- | Compute the kernel matrix.
  val kernel [n][m][o]: s -> [n][m]t -> [o][m]t -> [n]t -> [o]t -> *[n][o]t
}

module kernel_util (F: float) (P: {type s}) = {
  type t = F.t
  type s = P.s

  -- | Squared euclidean distance.
  let sqdist [n] (u: [n]t) (v: [n]t): t =
    F.(sum (map (\x -> x * x) (map2 (-) u v)))

  -- | Vector dot product.
  let dot [n] (u: [n]t) (v: [n]t): t =
    F.sum (map2 (F.*) u v)

  -- | Extract the diagonal of a full kernel matrix.
  let extdiag [n] (K: [n][n]t): *[n]t =
    map (\i -> K[i, i]) (iota n)

  -- | Compute diagonal of a kernel matrix for samples in X.
  let diag [n][m] (k: f t s [m]) (p: s) (X: [n][m]t): *[n]t =
    map (\u -> k p u u) X

  -- | Compute a full kernel matrix row.
  let row [n][m] (k: f t s [m]) (p: s) (X: [n][m]t)
      (v: [m]t) (_: [n]t) (_: t): *[n]t =
    map (\u -> k p u v) X

  let kernel [n][m][o] (k: f t s [m]) (p: s) (X0: [n][m]t)
      (X1: [o][m]t) (_: [n]t) (_: [o]t): *[n][o]t =
    map (\u -> map (k p u) X1) X0
}

-- | The linear kernel.
module linear (F: float): kernel = {
  type t = F.t
  type s = ()

  module P = {
    type s = s
  }

  module T = kernel_util F P

  let value _   = T.dot
  let extdiag _ = T.extdiag
  let diag      = T.diag   value
  let row       = T.row    value
  let kernel    = T.kernel value
}

-- | The simple version of rbf with no caching.
module rbf (F: float): kernel = {
  module T = kernel_util F {
    type s = {
      gamma: F.t
    }
  }

  type t = F.t
  type s = T.s

  let value [n] ({gamma}: s) (u: [n]t) (v: [n]t): t =
    F.exp (F.negate gamma F.* T.sqdist u v)

  let row p X0 u D0 d_u =
    map2 (\v d_v -> F.(d_v + d_u - i32 2 * T.dot u v)) X0 D0

  let kernel [n][m][o] (p: s) (X0: [n][m]t) (X1: [o][m]t)
      (D0: [n]t) (D1: [o]t): *[n][o]t =
    map (\_ ->(map (\_-> F.i32 0) X1)) X0

  let extdiag [n] _ (_: [n][n]t): *[n]t =
    replicate n (F.i32 1)

  let diag [n][m] _ (_: [n][m]t): *[n]t =
    replicate n (F.i32 1)
}
