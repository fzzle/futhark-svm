import "helpers"

entry solve [n_samples][n_features][n_as]
    (xs: [n_samples][n_features]f32)
    (fs: [n_as]bool) (starts: [n_as]i32) (cfs: [n_as]bool)
    (ys_s: [n_as]i8): ([]f32, i32, i32) =
  -- Parameters
  let C = 10
  let max_iterations = 100000

  --  i32.max 10000000 <|
  --    if n_samples > i32.highest / 100
  --    then i32.highest
  --    else 100 * n_samples

  let K = map (\ x -> map (\ x' -> dot x x') xs) xs
  let D = map (\ x -> f32.sum (map (\x_i -> x_i * x_i) x)) xs

  let A = replicate n_as 0f32
  let F = map (\y -> f32.i8 (-y)) ys_s
  let P = map (>0) ys_s

  let I = map2 (+) starts (segmented_iota cfs)
  let I_s = iota n_as

  let SF = scan (+) 0 (map i32.bool fs)
  let n_s = last SF
  
  -- Segment ends.
  let SE = rotate 1 fs
  -- Scan flags (segment index) at segment ends.
  let SFE = map2 (\f sf -> if f then sf - 1 else -1) SE SF
  let SF = map (\sf -> sf - 1) SF

  let (_, k, _, A) = loop (b, k, F, A) = (true, 0, F, A) while b do
    let F_u_B = map2 (\y a -> (y && a < C) || (!y && a > 0)) P A
    let F_u_I = map4 (\b f i i_s -> if b then (f, i, i_s) else (f32.inf, -1, -1)) F_u_B F I I_s
    let t0 = segmented_reduce (\a b -> if a.0 < b.0 then a else b) (f32.inf, -1, -1) n_s fs SFE F_u_I
    let (F_u, U, U_s) = unzip3 t0

    let F_l_B = map2 (\y a -> (y && a > 0) || (!y && a < C)) P A
    let F_l = map2 (\b f -> if b then f else -f32.inf) F_l_B F
    let F_max = segmented_reduce (\a b -> if a > b then a else b) (-f32.inf) n_s fs SFE F_l

    let t1 = map2 (\f_u f_max -> f_max - f_u < eps) F_u F_max
    in if all id t1 then (false, k, F, A) else

    let V_l_I = map4 (\f i i_s sf ->
      let (f_u, u, _) = t0[sf]
      let b = f_u - f
      -- If u is -1 then f_u is f32.inf and b is f32.inf.
      -- If f is not in F_lower then it's -f32.inf and b is f32.inf.
      -- Checks if i isnt -1 and f in F_lower and b < 0.
      in if b < 0
         then (b * b / (D[u] + D[i] - 2 * K[u, i]), i, i_s)
         else (-f32.inf, -1, -1)) F_l I I_s SF
    let t2 = segmented_reduce (\a b -> if a.0 > b.0 then a else b) (-f32.inf, -1, -1) n_s fs SFE V_l_I
    let (_, L, L_s) = unzip3 t2

    -- Find new A_u and A_l.
    let t3 = map3 (\(f_u, u, u_s) l l_s ->
      let y_u = f32.i8 ys_s[u_s]
      let y_l = f32.i8 ys_s[l_s]
      let b = f_u - F[l_s]
      
      let eta = D[u] + D[l] - 2 * K[u, l]

      let a_l = clamp (A[l_s] + y_l * b / eta) 0 C
      let a_u = clamp (A[u_s] + y_l * y_u * (A[l_s] - a_l)) 0 C

      let d_u = (a_u - A[u_s]) * y_u
      let d_l = (a_l - A[l_s]) * y_l
      in (a_u, a_l, d_u, d_l)) t0 L L_s
    let (A_u, A_l, _, _) = unzip4 t3

    let F = map3 (\f sf i ->
      let (_, _, d_u, d_l) = t3[sf]
      in f + d_u * K[U[sf], i] + d_l * K[L[sf], i]) F SF I

    let A = scatter A U_s A_u
    let A = scatter A L_s A_l

    in (k < max_iterations, k + 1, F, A)

  -- # support vectors
  let coefs = filter (\a -> f32.abs a > eps) A

  -- Todo: Find rho
  in (coefs, length coefs, k)
