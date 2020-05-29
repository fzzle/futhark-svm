import "helpers"

entry solve [n][m] (xs: [n][m]f32) (ys: [n]i8): ([]f32, f32, i32, i32) =
  let C = 10

  let gamma = 1 / (f32.i32 m)
  let degree = 3
  let coef0 = 0
  let max_iterations = 100000

  let I = iota n
  -- Q[i, j] = y[i] * y[j] * K[i, j]
  --let K = map (\x -> map (\x' -> dot x x') xs) xs
  let K = map (\x0 -> map (\x1 -> ((dot x0 x1) * gamma + coef0) ** degree) xs) xs
  let D = map (\i -> K[i, i]) I

  let A = replicate n 0f32
  let F = map (\y -> f32.i8 (-y)) ys
  let P = map (>0) ys
  
  let (_, k, F, A) = loop (c, k, F, A) = (true, 0, F, A) while c do
    -- Find the extreme example x_u, which has the minimum
    -- optimality indicator, f_u.
    let F_u_B = map2 (\y a -> (y && a < C) || (!y && a > 0)) P A
    let F_u_I = map3 (\b f i -> if b then (f, i) else (f32.inf, -1)) F_u_B F I
    let (f_u, u) = reduce_comm (\a b -> if a.0 < b.0 then a else b) (f32.inf, -1) F_u_I

    -- Find f_max so we can check if we're done. 
    let F_l_B = map2 (\y a -> (y && a > 0) || (!y && a < C)) P A
    let F_l = map2 (\b f -> if b then f else -f32.inf) F_l_B F
    let f_max = reduce_comm (\a b -> if a > b then a else b) (-f32.inf) F_l

    -- Check if done.
    in if f_max - f_u < eps then (false, k, F, A) else

    -- Find the extreme example x_l.
    let V_l_I = map4 (\f d k_u i ->
      let b = f_u - f
      -- f_u < f and f in X_l.
      -- b > 0 if f not in X_l (since f = -inf). 
      in if b < 0
         then ((b * b) / (D[u] + d - 2 * k_u), i)
         else (-f32.inf, -1)) F_l D K[u] I
    let (_, l) = reduce_comm (\a b -> if a.0 > b.0 then a else b) (-f32.inf, -1) V_l_I
    
    -- Find new a_u and a_l.
    let y_u = f32.i8 ys[u]
    let y_l = f32.i8 ys[l]
    let b = f_u - F[l]
    -- eta always > 0
    let eta = D[u] + D[l] - 2 * K[u, l]

    let a_l = clamp (A[l] + y_l * b / eta) 0 C
    let a_u = clamp (A[u] + y_l * y_u * (A[l] - a_l)) 0 C

    let d_u = (a_u - A[u]) * y_u
    let d_l = (a_l - A[l]) * y_l

    -- Update optimality indicators.
    let F = map3 (\f k_u k_l -> f + d_u * k_u + d_l * k_l) F K[u] K[l]
    
    let A[u] = a_u
    let A[l] = a_l

    in (k < max_iterations, k + 1, F, A)

  let objective_value = (reduce (+) 0 (map2 (*) A (map (\f -> f - 1) F))) / 2

  --let A = map2 (\a y -> a * f32.i8 y) A ys

  let coefs = filter (\a -> f32.abs a > eps) A

  -- Todo: Find rho
  in (coefs, objective_value, k, length coefs)
