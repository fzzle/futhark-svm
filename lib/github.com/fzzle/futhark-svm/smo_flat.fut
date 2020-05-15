import "helpers"

--     ys (input data):
--     +---+---------+-----+
--   c | 1 |    2    |  3  |
--     +-+-+-+-+-+-+-+-+-+-+
--  ys |1|1|2|2|2|2|2|3|3|3|
--     +-+-+-+-+-+-+-+-+-+-+

--     Class pair ys (for flattened solve):
--     +---+---------+-----+-----+---------+-----+
--   c | 1 |    2    |  1  |  3  |    2    |  3  |
--     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
--  ys |0|0|1|1|1|1|1|0|0|0|1|1|1|0|0|0|0|0|1|1|1|
--     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
--  fs |1|0|0|0|0|0|0|1|0|0|0|0|0|1|0|0|0|0|0|0|0|
--     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

-- Differences from smo_parallel:
-- 1. Can't use Q-matrix since non-flattened ys aren't -1 / +1.
--    Instead I use a K-matrix (kernel) and multiply by the y's
--    when needed later. However, this makes it unecessary to
--    multiply by y's when computing new A's (it's only needed
--    when computing new G's).

entry solve [n_samples][n_features][n_as]
    (xs: [n_samples][n_features]f32)
    (fs: [n_as]bool) (starts: [n_as]i32) (cfs: [n_as]bool)
    (aligned_ys: [n_as]i8): ([n_as]f32, i32) =
  -- Parameters
  let C = 10
  let max_iterations = 1000

  --  i32.max 10000000 <|
  --    if n_samples > i32.highest / 100
  --    then i32.highest
  --    else 100 * n_samples

  -- Example:
  -- ys:         [0, 0, 0, 1, 1, 1]
  -- starts:     [0, 0, 0, 3, 3, 3]
  -- aligned_ys: [0, 0, 0, 1, 1, 1]
  -- fs:         [1, 0, 0, 0, 0, 0]

  -- Q[i, j] = y[i] * y[j] * K[i, j]
  let K = map2 (\ x -> map2 (\ x' -> dot x x') xs) xs
  let D = map (\ x -> f32.sum (map (\ x_i -> x_i * x_i) x)) xs

  let A = replicate n_as 0f32
  let G = replicate n_as (-1f32)
  let stops = replicate n_as false

  let segmented_indices = map2 (+) starts (segmented_iota c_fs)
  let fys = map (==1) aligned_ys

  let scan_flags = scan (+) 0 (map i32.bool fs)
  let n_segments = last scan_flags
  
  let (_, i, _, A, _) = loop (b, i, G, A, s) = (true, 0, G, A, stops) while b do
    let bcs0 = map3 (\s y a -> !s && ((y && a < C) || (!y && a > 0))) s fys A
    let bcs1 = map3 (\s y a -> !s && ((y && a > 0) || (!y && a < C))) s fys A
    let tmp0 = map2 (\y g -> f32.i8 (-y) * g) aligned_ys G
    --let Gxs = map2 (\b t -> if b then t else f32.nan) bcs0 tmp0
    --let Gns = map2 (\b t -> if b then t else f32.nan) bcs1 tmp0

    -- Find G_maxs + their is, as (potentially ys)
    let tmp = segmented_scan (\a b -> if !a.1 || (b.1 && b.0 >= a.0) then b else a)
      (-f32.inf, true, -1, -1, 0) fs (zip5 tmp0 bcs0 segmented_indices (iota n_as) A)
    
    let (G_maxs', _, is', iss', as') = unzip5 tmp

    let s_G_maxs = f32_extract_endings n_segments fs G_maxs'
    let s_is     = i32_extract_endings n_segments fs is'
    let s_A_is   = f32_extract_endings n_segments fs as'
    let s_iss    = i32_extract_endings n_segments fs iss'

    let d_G_maxs = f32_distribute_endings fs s_G_maxs'
    let d_is     = i32_distribute_endings fs s_is'

    let (G_mins', _) = unzip <| segmented_scan (\ a b ->
      if !a.1 || (b.1 && b.0 <= a.0) then b else a)
      (f32.inf, true) fs (zip tmp0 bcs1)

    let d_G_mins = f32_distribute_endings fs G_mins'
    let s_G_mins = f32_extract_endings n_segments fs G_mins'

    let bs = map3 (\ g y gx -> gx + (f32.i8 y) * g) G aligned_ys d_G_maxs
    let bcs2 = map2 (\ b c -> b > 0 && c) bs bcs0

    let Ons = map3 (\ b i t ->
      let a = D[i] + D[t] - 2 * K[i, t]
      let a = f32.max a tau
      in -(b * b) / a)
      bs d_is s_iss segmented_indices

    let (_, _, s_js, s_jss, s_A_js) = unzip5 (segmented_reduce (\ a b ->
      if !a.1 || (b.1 && b.0 <= a.0) then b else a)
      (f32.inf, true, -1, -1, 0) fs (zip5 Ons bcs2 segmented_indices (iota n_as) A))

    let s_js = s_js :> [n_segments]i32
    let s_A_js = s_A_js :> [n_segments]f32
    let s_jss = s_jss :> [n_segments]i32

    let (s_is, s_js) = unzip <| map4 (\ i j Gx Gn -> if j == -1 || Gx - Gn < eps then (-1, -1) else (i, j)) s_is s_js s_G_maxs s_G_mins
    let s_s' = map (==(-1)) s_js
    in if all id s_s' || i >= max_iterations then (false, i, G, A, s) else

    let newAs = map5 (\ i j curr_A_i curr_A_j (is, js)  ->
      if j == -1 then (0, 0, 0, 0) else
        let y_if = f32.i8 aligned_ys[is]
        let y_jf = f32.i8 aligned_ys[js]

        let a = D[i] + D[j] - 2f32 * K[i, j]
        let a = f32.max a tau
        let b = (-y_if) * G[is] + y_jf * G[js]

        -- update alphas
        let A_i = curr_A_i + y_if * b / a
        let sum = y_if * curr_A_i + y_jf * curr_A_j
        let A_i = clamp A_i 0 C
        let A_j = y_jf * (sum - y_if * A_i)
        let A_j = clamp A_j 0 C
        let A_i = y_if * (sum - y_jf * A_j)
        in (A_i, A_j, A_i - curr_A_i, A_j - curr_A_j)
      ) s_is s_js s_A_is s_A_js (zip2 s_iss s_jss)

    let (A_is, A_js, delta_A_is, delta_A_js) = unzip4 newAs
    
    let d_dA_is = f32_distribute fs delta_A_is
    let d_dA_js = f32_distribute fs delta_A_js
    let d_js = i32_distribute fs s_js

    let G' = map5 (\ g i j t (dA_i, dA_j) -> g + Q[i, t] * dA_i + Q[j, t] * dA_j)
      G d_is d_js segmented_indices (zip d_dA_is d_dA_js) 

    let n_segmentsx2 = n_segments * 2
    let idxs = (s_iss ++ s_jss) :> [n_segmentsx2]i32
    let vs = (A_is ++ A_js) :> [n_segmentsx2]f32
    let A' = scatter (copy A) idxs vs

    let s' = map4 (\ s j Gx Gn -> s || j == -1 || Gx - Gn < eps) s d_js d_G_maxs d_G_mins
    in (true, i + 1, G', A', s')

  -- # support vectors
  --let n_sv = reduce (\ c a -> c + i32.bool (f32.abs a > 0)) 0 A

  -- Todo: Find rho
  in (A, i)
