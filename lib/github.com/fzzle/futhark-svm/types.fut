-- | Model settings type.
type training_settings 't = {
  -- Working set size.
  n_ws: i64,
  -- Max iterations to be performed by the solver. For the solver
  -- using two-level decomposition, this is the max number of total
  -- inner iterations allowed. -1 = infinite.
  max_t: i64,
  -- Max number of inner iterations.
  max_t_in: i64,
  -- Max number of outer iterations.
  max_t_out: i64,
  -- Tolerance.
  eps: t
}

type prediction_settings 't = {
  -- Prediction batch size.
  n_ws: i64
}

type weights 't [m][o][p][q] = {
  -- Flat alphas.
  A: [o]t,
  -- Flat support vector indices.
  I: [o]i64,
  -- Support vectors.
  S: [p][m]t,
  -- Segment sizes of flat alphas/indices.
  Z: [q]i64,
  -- Rhos (-bias/intercept).
  R: [q]t,
  -- Number of classes.
  n_c: i64
}

type details 't [q] = {
  -- Objective values.
  O: [q]t,
  -- Total iterations (inner).
  T: [q]i64,
  -- Outer iterations.
  T_out: [q]i64
}

-- | Trained model type.
type output 't [m][o][p][q] = {
  weights: weights t [m][o][p][q],
  details: details t [q]
}
