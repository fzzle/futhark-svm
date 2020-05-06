import "helpers"
import "flatten"

entry solve [n][m] (xs: [n][m]f32) (ys: [n]i8): [n]f32 =
  let C = 10
  
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

  let size = n * (n - 1) / 2
  let A = replicate size 0f32
  let G = replicate size (-1f32)
  --let indices = iota n
  -- Segmented indices that still point to the samples
  -- contained in `ys`, `xs`, and the cache `Q`, `D`.
  let segmented_indices = map2 (+) starts (segmented_iota fs)

  let (_, _, A) = loop (c, G, A) = (true, G, A) while c do
    -- working set selection 3
    let mxss = map3 (\ a g y -> if (y == 1 && a < C) ||
      (y == -1 && a > 0) then f32.i8 (-y) * g else f32.nan) A G aligned_ys

    -- segmented_reduce_distribute
    let (is, mxs) = unzip (segmented_scan (\ (i, x0) (t, x1) ->
      if !(f32.isnan x1) && x1 >= x0 then (t, x1) else (i, x0))
      (-1, -f32.inf) fs (zip segmented_indices Gxss))
    let is = distribute_ends fs is
    let mxs = distribute_ends fs mxs


    let (is, Gxs) = unzip (segmented_reduce (\ (i, Gx) (t, Gx') ->
      if !(f32.isnan Gx') && Gx' >= Gx then (t, Gx') else (i, Gx))
      (-1, -f32.inf) fs (zip segmented_indices Gxss))

    -- Flatten: Replace A with As and ys with yss 
    let cs = map2 (\ a y -> (y == 1 && a > 0) || (y == -1 && a < C)) A aligned_ys
    -- Flatten: Insert Gs yss css
    -- Would be kind of bad, but (f32.i8 (-y) * g) / f32.bool c should return the same
    let Gnss = map3 (\ g y c -> if c then f32.i8 (-y) * g else f32.nan) G aligned_ys cs
    -- Flatten by segmented reduce w/ flags
    let Gns = segmented_reduce (\ Gn Gn' ->
      if !(f32.isnan Gn') && Gn' <= Gn then Gn' else Gn)
      f32.inf fs Gnss
    
    let y_if = f32.i8 ys[i]
    let q_i = Q[i]
    let d_i = D[i]

    -- Flatten by map
    let distributed_gxs = distribute fs Gxs 
    let bs = map3 (\ g y gx -> gx + (f32.i8 y) * g) G aligned_ys distributed_gxs
    
    let as = map3 (\ q d y -> 
      let a = d_i + d - 2f32 * y_if * (f32.i8 y) * q
      in f32.max a tau) q_i D ys
    let Ons = map3 (\ c b a -> if c && b > 0 then -(b * b) / a else f32.nan) cs bs as
    let (j, _) = reduce (\ (j, On) (t, On') -> if !(f32.isnan On') && On' <= On
      then (t, On') else (j, On)) (-1, f32.inf) (zip iots Ons)

    in if j == -1 || Gx - Gn < eps then (false, G, A) else

    let y_jf = f32.i8 ys[j]

    -- working set: (i, j)
    let a = as[j]
    let b = bs[j]

    -- update alphas
    let A_i = A[i] + y_if * b / a
    -- let A_j = A[j] - f32.i8 ys[j] * b / a
    let sum = y_if * A[i] + y_jf * A[j]
    let A_i = clamp A_i 0 C
    let A_j = y_jf * (sum - y_if * A_i)
    let A_j = clamp A_j 0 C
    let A_i = y_if * (sum - y_jf * A_j)

    -- update gradient
    let dA_i = A_i - A[i]
    let dA_j = A_j - A[j]
    let G' = map3 (\ q_i q_j g -> g + q_i * dA_i + q_j * dA_j) Q[i] Q[j] G
    let A' = scatter A [i, j] [A_i, A_j]

    in (true, G', A')

  -- # support vectors
  --let n_sv = reduce (\ c a -> c + i32.bool (f32.abs a > 0)) 0 A

  -- Todo: Find rho
  in A
