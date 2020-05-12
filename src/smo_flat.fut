import "helpers"
import "flatten"

entry solve [n_samples][n_features][n_as]
    (xs: [n_samples][n_features]f32) (ys: [n_samples]i8)
    (fs: [n_as]bool) (starts: [n_as]i32) (c_fs: [n_as]bool)
    (aligned_ys: [n_as]i8): ([n_as]f32, i32) =
  let C = 10
  let max_iterations = 10000
  -- Todo: starts, an array of flag values that indicate
  -- the start index of a class group in `ys`.
  -- Todo: algined_ys, an array of 0 and 1's that indicate
  -- the related class of each A-value.
  -- Todo: fs, a flag array representing start of pair.

  -- Example:
  -- ys:         [0, 0, 0, 1, 1, 1]
  -- starts:     [0, 0, 0, 3, 3, 3]
  -- algined_ys: [0, 0, 0, 1, 1, 1]
  -- fs:         [1, 0, 0, 0, 0, 0]

  -- Q[i, j] = y[i] * y[j] * K[i, j]
  let Q = map2 (\ x y -> map2 (\ x' y' -> f32.i8 (y * y') * dot x x') xs ys) xs ys
  let D = map (\ x -> f32.sum (map (\ x_i -> x_i * x_i) x)) xs

  let A = replicate n_as 0f32
  let G = replicate n_as (-1f32)
  --let indices = iota n
  -- Segmented indices that still point to the samples
  -- contained in `ys`, `xs`, and the cache `Q`, `D`.

  let segmented_indices = map2 (+) starts (segmented_iota c_fs)
  let end_flags = rotate 1 fs
  let reversed_end_flags = reverse end_flags
  let y_flags = map (==1) aligned_ys

  let scan_flags = scan (+) 0 (map i32.bool fs)
  let n_segments = last scan_flags

  let (_, i, G, A) = loop (c, i, G, A) = (true, 0, G, A) while c do
    let bcs0 = map2 (\y a -> (y && a < C) || (!y && a > 0)) y_flags A
    let bcs1 = map2 (\y a -> (y && a > 0) || (!y && a < C)) y_flags A
    let tmp0 = map2 (\y g -> f32.i8 (-y) * g) aligned_ys G
    let Gxs = map2 (\b t -> if b then t else f32.nan) bcs0 tmp0
    let Gns = map2 (\b t -> if b then t else f32.nan) bcs1 tmp0

    -- Find G_maxs + their is, as (potentially ys)
    let tmp = segmented_scan (\(x0, t0, u0, a0) (x1, t1, u1, a1) ->
      let next = f32.isnan x0 || (!(f32.isnan x1) && x1 >= x0)
      in if next then (x1, t1, u1, a1) else (x0, t0, u0, a0))
      (-f32.inf, -1, -1, 0) fs (zip4 Gxs segmented_indices (iota n_as) A)
    
    let (G_maxs', is', iss', as') = unzip4 tmp
    
    let d_G_maxs = f32_distribute_endings fs G_maxs'
    let d_is     = i32_distribute_endings fs is'

    let s_G_maxs = f32_extract_endings n_segments fs G_maxs'
    let s_is     = i32_extract_endings n_segments fs is'
    let s_A_is   = f32_extract_endings n_segments fs as'
    let s_iss    = i32_extract_endings n_segments fs iss'

    -- in if true then (false, i32.f32 d_G_maxs[0], G, A) else

    let s_G_mins = segmented_reduce (\ x0 x1 ->
      if f32.isnan x0 || !(f32.isnan x1) && x1 <= x0 then x1 else x0)
      f32.inf fs Gns
      :> [n_segments]f32

    let bs = map3 (\ g y gx -> gx + (f32.i8 y) * g) G aligned_ys d_G_maxs    
    let as = map2 (\ i t -> 
      -- if i == -1 then 0, just to make -(b * b) / a = nan, and thus ignored in the reduce
      let a = D[i] + D[t] - 2f32 * f32.i8 (ys[i] * ys[t]) * Q[i, t]
      in f32.max a tau) d_is segmented_indices
    let Ons = map3 (\ c b a -> if c && b > 0 then -(b * b) / a else f32.nan) bcs1 bs as

    let (_, s_js, s_jss, s_A_js) = unzip4 (segmented_reduce (\ (x0, t0, u0, a0) (x1, t1, u1, a1) ->
      let next = f32.isnan x0 || (!(f32.isnan x1) && x1 <= x0)
      in if next then (x1, t1, u1, a1) else (x0, t0, u0, a0))
      (f32.inf, -1, -1, 0) fs (zip4 Ons segmented_indices (iota n_as) A))

    let s_js = s_js :> [n_segments]i32
    let s_A_js = s_A_js :> [n_segments]f32
    let s_jss = s_jss :> [n_segments]i32

    let stops = map3 (\ j Gx Gn -> j == -1 || (Gx - Gn < eps)) s_js s_G_maxs s_G_mins
    in if all id stops || i > max_iterations then (false, i, G, A) else

    let newAs = map5 (\ i j curr_A_i curr_A_j s ->
      if s then (0, 0, 0, 0) else
        let y_if = f32.i8 ys[i]
        let y_jf = f32.i8 ys[j]

        let a = D[i] + D[j] - 2f32 * y_if * y_jf * Q[i, j]
        let a = f32.max a tau
        let b = (-y_if) * G[i] + y_jf * G[j] --forkert

        -- update alphas
        let A_i = curr_A_i + y_if * b / a
        let sum = y_if * curr_A_i + y_jf * curr_A_j
        let A_i = clamp A_i 0 C
        let A_j = y_jf * (sum - y_if * A_i)
        let A_j = clamp A_j 0 C
        let A_i = y_if * (sum - y_jf * A_j)
        in (A_i, A_j, A_i - curr_A_i, A_j - curr_A_j)
      ) s_is s_js s_A_is s_A_js stops

    let (A_is, A_js, delta_A_is, delta_A_js) = unzip4 newAs
    
    let d_dA_is = distribute fs delta_A_is
    let d_dA_js = distribute fs delta_A_js
    let d_js = i32_distribute fs s_js

    let G' = map5 (\ g i j t (dA_i, dA_j) -> g + Q[i, t] * dA_i + Q[j, t] * dA_j)
      G d_is d_js segmented_indices (zip d_dA_is d_dA_js) 

    -- forkert (is og js er globale indeks)
    -- let wb_is =  ++ s_js :> [n_segments_x2]i32
    -- let wb_As =  ++ A_js :> [n_segments_x2]f32
    let A' = scatter (copy A) s_iss A_is
    let A' = scatter A' s_jss A_js

    in (true, i + 1, G', A')

  -- # support vectors
  --let n_sv = reduce (\ c a -> c + i32.bool (f32.abs a > 0)) 0 A

  -- Todo: Find rho
  in (A, i)
