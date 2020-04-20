


let dot [n] (as: [n]f64) (bs: [n]f64): f64 =
  f64.sum (map2 (*) as bs)

let clamp (x: f64) (min: f64) (max: f64): f64 =
  if x < min then min else if x > max then max else x

entry train_binary [n][m] (xs: [n][m]f64) (ys: [n]i8): ([n]f64) =
  let C = 10
  -- let p_tol = 0.001
  -- let p_max_passes = 100
  -- internal:
  let eps = 1e-3
  let tau = 1e-6 ---12

  -- let K = map (\ x -> map (\ x' -> dot x x') xs) xs
  let Q = map2 (\ x y -> map2 (\ x' y' -> f64.i8 (y * y') * dot x x') xs ys) xs ys
  -- Find hurtigere y * y'
  -- todo: Symmetric, so we have duplicate values
  let A = replicate n 0f64
  let G = replicate n (-1f64)

  -- let s = (A, G)
  let (_, A, _) = loop (c, A, G) = (true, A, G) while c do
    -- selectB
    let G_max = f64.lowest
    let G_min = f64.highest
    let (i, G_max) = loop (i, G_max) = (-1, G_max) for t < n do
      let y_t = ys[t]
      let A_t = A[t]
      let y_tf = f64.i8 y_t
      let cond0 = (y_t == 1 && A_t < C) || (y_t == -1 && A_t > 0)
      let cond1 = -y_tf * G[t] >= G_max
      in if cond0 && cond1 then (t, -y_tf * G[t]) else (i, G_max)

    let y_if = f64.i8 ys[i]
    let obj_min = f64.highest
    let (j, _, G_min) = loop (j, obj_min, G_min) = (-1, obj_min, G_min) for t < n do
      let y_t = ys[t]
      let A_t = A[t]
      let y_tf = f64.i8 y_t
      let cond0 = (y_t == 1 && A_t > 0) || (y_t == -1 && A_t < C)
      let cond1 = (-y_tf) * G[t] <= G_min
      in if !cond0 then (j, obj_min, G_min) else
      let b = G_max + y_tf * G[t]
      let G_min' = (if cond1 then -y_tf * G[t] else G_min)
      in if !(b > 0) then (j, obj_min, G_min') else
      let a = Q[i, i] + Q[t, t] - 2f64 * y_if * y_tf * Q[i, t]
      let a = (if a <= 0 then tau else a)
      let cond2 = -(b * b) / a <= obj_min
      in if cond2 then (t, -(b * b) / a, G_min') else (j, obj_min, G_min')

    let cond0 = G_max - G_min < eps
    in if cond0 || j == -1 then (false, A, G) else

    --      QDia[i] + QDia[j] -
    let a = Q[i, i] + Q[j, j] - 2 * f64.i8 (ys[i] * ys[j]) * Q[i, j]
    let a = if a <= 0 then tau else a
    let b = f64.i8 (-ys[i]) * G[i] + f64.i8 ys[j] * G[j]

    -- Update alpha
    --let oldAi = A[i]
    --let oldAj = A[j]
    let A_i = A[i] + f64.i8 ys[i] * b / a
    --let A_j = A[j] - f64.i8 ys[j] * b / a
    let sum = f64.i8 ys[i] * A[i] + f64.i8 ys[j] * A[j]
    let A_i = clamp A_i 0 C
    let A_j = f64.i8 ys[j] * (sum - f64.i8 ys[i] * A_i)
    let A_j = clamp A_j 0 C
    let A_i = f64.i8 ys[i] * (sum - f64.i8 ys[j] * A_j)

    -- Update gradient
    let deltaAi = A_i - A[i]
    let deltaAj = A_j - A[j]
    let G' = map2 (\ q g -> g + q[i] * deltaAi + q[j] * deltaAj) Q G
    let A' = scatter (copy A) [i, j] [A_i, A_j]
    --A with [i] = A_i -- scatter
    --let A' = A' with [j] = A_j 
    in (true, A', G')
  --let (A, _) = s
  in A

--let main [n] (xs: f64[m][n]) (ys: f64[n]): ([n]f64, f64) =


-- let predict [n][m] (xs: [n][m]f64): [n]i16 =
  -- predict: model -> xs -> ys
  -- replicate n 0i16
