import "helpers"



entry solve [n_samples][n_features][n_ys]
    (xs: [n_samples][n_features]f32) (fs: [n_ys]bool)
    (starts: [n_ys]i32) (cfs: [n_ys]bool) (ys: [n_ys]bool): () =

  let C = 10
  let max_iterations = 100000

  let is_upper (p: bool) (a: f32): bool =
    (p && a < C) || (!p && a < 0)

  let is_lower (p: bool) (a: f32): bool =
    (p && a > 0) || (!p && a < C)

  -- Find the linear kernel matrix
  let K = map (\x -> map (\ x' -> dot x x') xs) xs
  let D = map (\x -> f32.sum (map (\x_i -> x_i * x_i) x)) xs

  let A = replicate n_as 0f32
  let F = map (\y -> f32.i8 (-y)) ys_s

  let I = map2 (+) starts (segmented_iota cfs)
  let I_s = iota n_as

  loop (bo, ko, Fe, Ae, F, A, fs, ys) = (true, 0, [], [], F, A, fs, ys) while bo do
    let segment_offsets = map i32.bool fs |> scan (+) 0 |> map (\i -> i - 1)
    let n_segments = last segment_offsets - 1
    let mask i f = if f then i else -1
    let segment_end_offsets = map2 mask segment_offsets (rotate 1 fs)

    let P = map (>0) ys
    let I_s = iota n_as

    -- tmp
    let n_s = n_segments
    let SF = segment_offsets
    let SFE = segment_end_offsets

    let (_, ki, A, F, done) = loop (bi, ki, A, F, done) = (true, ko, A, F, []) while bi do
      let F_u_B = map2 is_upper P A
      let F_u_I = map4 (\b f i i_s -> if b then (f, i, i_s) else (f32.inf, -1, -1)) F_u_B F I I_s
      let t0 = segmented_reduce (\a b -> if a.0 < b.0 then a else b) (f32.inf, -1, -1) n_s fs SFE F_u_I
      let (F_u, U, U_s) = unzip3 t0

      let F_l_B = map2 is_lower P A
      let F_l = map2 (\b f -> if b then f else -f32.inf) F_l_B F
      let F_max = segmented_reduce (\a b -> if a > b then a else b) (-f32.inf) n_s fs SFE F_l

      let t1 = map2 (\f_u f_max -> f_max - f_u < eps) F_u F_max

      in if any id t1 then (false, k, F, A, t1) else

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

      in (k < max_iterations, k + 1, F, A, done)

    let done = map (\i -> done[i]) segment_offsets
    let (finished, continued) = partition (.0) (zip A F )


    filter (\ -> ) flags segment_indices A F ys



    in ()


    -- # support vectors
    let coefs = filter (\a -> f32.abs a > eps) A

  -- Todo: Find rho
  in (coefs, length coefs, k)
