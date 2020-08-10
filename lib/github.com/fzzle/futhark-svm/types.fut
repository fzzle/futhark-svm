-- | Model settings type.
type training_settings 't = {
  -- Working set size.
  n_ws: i32,
  -- Max iterations to be performed by the solver. For the solver
  -- using two-level decomposition, this is the max number of total
  -- inner iterations allowed. -1 = infinite.
  max_t: i32,
  -- Max number of inner iterations.
  max_t_in: i32,
  -- Max number of outer iterations.
  max_t_out: i32,
  -- Tolerance.
  eps: t
}

type prediction_settings 't = {
  -- Prediction batch size.
  n_ws: i32
}

type~ weights 't [m] = {
  -- Flat alphas.
  A: []t,
  -- Flat support vector indices.
  I: []i32,
  -- Support vectors.
  S: [][m]t,
  -- Segment sizes of flat alphas/indices.
  Z: []i32,
  -- Rhos (-bias/intercept).
  R: []t,
  -- Number of classes.
  n_c: i32
}

type~ details 't = {
  -- Objective values.
  O: []t,
  -- Total iterations (inner).
  T: []i32,
  -- Outer iterations.
  T_out: []i32
}

-- | Trained model type.
type~ output 't [m] = {
  weights: weights t [m],
  details: details t
}
