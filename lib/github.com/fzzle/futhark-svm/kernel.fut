module type kernel_function = {
  -- | Kernel parameters.
  type s
  -- | Float type.
  type t
  -- | Compute a single kernel value.
  val value [n]: s -> [n]t -> [n]t -> t
}

-- | Kernel computation module type.
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
module kernel_util (R: real) = {
  type t = R.t
  -- | Value function type.
  type^ f [n] = [n]t -> [n]t -> t
  -- | Squared euclidean distance.
  let sqdist [n] (u: [n]t) (v: [n]t): t =
    R.(sum (map (\x -> x * x) (map2 (-) u v)))

  -- | Vector dot product.
  let dot [n] (u: [n]t) (v: [n]t): t =
    R.(sum (map2 (*) u v))

  -- | Extract the diagonal of a symmetric kernel matrix.
  let extdiag [n] (K: [n][n]t): *[n]t =
    map (\i -> K[i, i]) (iota n)

  -- | Compute diagonal of a kernel matrix for samples in X.
  let diag [n][m] (k: f [m]) (X: [n][m]t): *[n]t =
    map (\u -> k u u) X

  -- | Compute a full kernel matrix row.
  let row [n][m] (k: f [m]) (X: [n][m]t)
      (u: [m]t) (_: [n]t) (_: t): *[n]t =
    map (\v -> k u v) X

  let matrix [n][m][o] (k: f [m]) (X0: [n][m]t)
      (X1: [o][m]t) (_: [n]t) (_: [o]t): *[n][o]t =
    map (\u -> map (\v -> k u v) X1) X0
}

-- | Parametric module which makes it possible to implement a new
-- kernel type only by providing the kernel function and the type
-- for the model parameters of the function. Alternatively, it's
-- possible to define a kernel from scratch using the kernel module
-- type. It allows the specification of how the matrix, diag, etc.
-- should be computed, in case it can be done better than w/ this.
module default_kernel (R: real) (T: kernel_function with t = R.t) = {
  module util = kernel_util R

  type t = T.t
  type s = T.s

  let value = T.value
  let extdiag _     = util.extdiag
  let diag   (p: s) = util.diag   (value p)
  let row    (p: s) = util.row    (value p)
  let matrix (p: s) = util.matrix (value p)
}

-- | The linear kernel:
-- K(u, v) = <u, v>
module linear (R: real): kernel
    with t = R.t
    with s = {} = default_kernel R {
  module util = kernel_util R

  type t = R.t
  type s = {}

  let value _ = util.dot
}

-- | The sigmoid kernel:
-- K(u, v) = tanh (\gamma * <u, v> + c)
module sigmoid (R: real): kernel
    with t = R.t
    with s = {gamma: R.t, coef0: R.t} = default_kernel R {
  module util = kernel_util R

  type t = R.t
  type s = {
    gamma: t,
    coef0: t
  }

  let value [n] (p: s) (u: [n]t) (v: [n]t): t =
    R.(tanh (p.gamma * util.dot u v + p.coef0))
}

-- | The polynomial kernel:
-- K(u, v) = (\gamma * <u, v> + c) ** degree
module polynomial (R: real): kernel
    with t = R.t
    with s = {gamma:R.t,coef0:R.t,degree:R.t} = default_kernel R {
  module util = kernel_util R

  type t = R.t
  type s = {
    gamma: t,
    coef0: t,
    degree: t
  }

  let value [n] (p: s) (u: [n]t) (v: [n]t): t =
    R.((p.gamma * util.dot u v + p.coef0) ** p.degree)
}

-- | The rbf kernel with simple squared distance calculated by:
-- K(u, v) = \exp(-\gamma * ||u - v||²)
--         = \exp(-\gamma * (\sum (u_i - v_i)²))
module rbf (R: real): kernel
    with t = R.t
    with s = {gamma: R.t} = {
  module util = kernel_util R

  type t = R.t
  type s = {
    gamma: t
  }

  let value [n] ({gamma}: s) (u: [n]t) (v: [n]t): t =
    R.exp (R.neg gamma R.* util.sqdist u v)

  -- rbf diagonals are always all 1 because the squared distance
  -- of a point to itself is 0 and \exp(\gamma * 0) is 1.
  let extdiag [n] (_: s) (_: [n][n]t): *[n]t =
    replicate n (R.i32 1)

  let diag [n][m] (_: s) (_: [n][m]t): *[n]t =
    replicate n (R.i32 1)

  let row    (p: s) = util.row    (value p)
  let matrix (p: s) = util.matrix (value p)
}

-- | The rbf kernel where the matrix and row operations use
-- <u, u> + <v, v> - 2 * <u, v> to find the squared distance:
-- K(u, v) = \exp(-\gamma * (<u, u> + <v, v> - 2 * <u, v>))
-- It happens to be much slower than rbf in practice.
module rbf_pre (R: real): kernel
    with t = R.t
    with s = {gamma: R.t} = {
  module util = kernel_util R

  type t = R.t
  type s = {
    gamma: t
  }

  let value [n] ({gamma}: s) (u: [n]t) (v: [n]t): t =
    R.exp (R.neg gamma R.* util.sqdist u v)

  let extdiag [n] (_: s) (_: [n][n]t): *[n]t =
    replicate n (R.i32 1)

  let diag [n][m] (_: s) (_: [n][m]t): *[n]t =
    replicate n (R.i32 1)

  let row [n][m] ({gamma}: s) (X: [n][m]t)
      (u: [m]t) (D: [n]t) (d_u: t): *[n]t =
    let k x = R.exp (R.neg gamma R.* x)
    in map2 (\v d_v -> k R.(d_u + d_v - i32 2 * util.dot u v)) X D

  let matrix [n][m][o] (p: s) (X0: [n][m]t)
      (X1: [o][m]t) (D0: [n]t) (D1: [o]t): *[n][o]t =
    map2 (\u d_u -> row p X1 u D1 d_u) X0 D0
}

-- | Aggregation module for kernels.
module kernels (R: real) = {
  module linear     = linear R
  module sigmoid    = sigmoid R
  module polynomial = polynomial R
  module rbf        = rbf R
}
