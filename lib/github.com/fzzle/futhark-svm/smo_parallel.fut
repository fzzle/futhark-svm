import "helpers"

entry solve [n][m] (xs: [n][m]f32) (ys: [n]i8): ([]f32, i32) =
  let C = 10

  -- Q[i, j] = y[i] * y[j] * K[i, j]
  let Q = map2 (\ x y -> map2 (\ x' y' -> f32.i8 (y * y') * dot x x') xs ys) xs ys
  let D = map (\ x -> f32.sum (map (\ x_i -> x_i * x_i) x)) xs

  let A = replicate n 0f32
  let G = replicate n (-1f32)
  let iots = iota n

  -- let s = (A, G)
  let (_, _, A) = loop (c, G, A) = (true, G, A) while c do
    -- working set selection 3
    let Gxs = map3 (\ a g y -> if (y == 1 && a < C) ||
      (y == -1 && a > 0) then f32.i8 (-y) * g else f32.nan) A G ys
    -- Can use reduce_comm if we make the operator commutative.
    -- We can do that by comparing i and t first, then Gx' and Gx.
    let (i, Gx) = reduce (\ (i, Gx) (t, Gx') -> if !(f32.isnan Gx') && Gx' >= Gx
      then (t, Gx') else (i, Gx)) (-1, -f32.inf) (zip iots Gxs)

    let cs = map2 (\ a y -> (y == 1 && a > 0) || (y == -1 && a < C)) A ys
    let Gns = map3 (\ g y c -> if c then f32.i8 (-y) * g else f32.nan) G ys cs
    let Gn = reduce (\ Gn Gn' -> if !(f32.isnan Gn') && Gn' <= Gn then Gn' else Gn) f32.inf Gns
    
    let y_if = f32.i8 ys[i]
    let q_i = Q[i]
    let d_i = D[i]
    let bs = map2 (\ g y -> Gx + (f32.i8 y) * g) G ys
    let as = map3 (\ q d y -> 
      let a = d_i + d - 2f32 * y_if * (f32.i8 y) * q
      in f32.max a tau) q_i D ys
    let Ons = map3 (\ c b a -> if c && b > 0 then -(b * b) / a else f32.nan) cs bs as
    let (j, _) = reduce (\ (j, On) (t, On') -> if !(f32.isnan On') && On' <= On
      then (t, On') else (j, On)) (-1, f32.inf) (zip iots Ons)

    let c0 = Gx - Gn < eps
    let c1 = j == -1
    in if c1 || c0 then (false, G, A) else

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
  let is_sv = map (\a -> i32.bool (a != 0)) A
  let n_sv = i32.sum is_sv

  let A = map2 (\a y -> a * f32.i8 y) A ys

  let coefs = filter (\a -> a != 0) A

  -- Todo: Find rho
  in (coefs, n_sv)
