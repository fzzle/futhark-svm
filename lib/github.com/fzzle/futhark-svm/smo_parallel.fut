import "helpers"

entry solve [n][m] (xs: [n][m]f32) (ys: [n]i8): ([]f32, i32) =
  let C = 10
  let max_iterations = 100000

  -- Q[i, j] = y[i] * y[j] * K[i, j]
  let Q = map2 (\ x y -> map2 (\ x' y' -> f32.i8 (y * y') * dot x x') xs ys) xs ys
  let D = map (\ x -> f32.sum (map (\ x_i -> x_i * x_i) x)) xs

  let A = replicate n 0f32
  let G = replicate n (-1f32)
  let iots = iota n

  -- let s = (A, G)
  let (_, _, A, n) = loop (c, G, A, n) = (true, G, A, 0) while c do
    -- working set selection 3

    let Gx_is = map4
      (\y a g i -> if (y > 0 && a < C) || (y < 0 && a > 0)
      then (f32.i8 (-y) * g, i) else (-f32.inf, -1)) 
      ys A G iots

    let (Gx, i) = reduce_comm
      (\a b -> if b.0 >= a.0 then b else a)
      (-f32.inf, -1) Gx_is

    let Gn = map3 (\y a g ->
      if (y > 0 && a > 0) || (y < 0 && a < C)
      then f32.i8 (-y) * g else f32.inf)
      ys A G

    let Gn = reduce_comm
      (\a b -> if b <= a then b else a)
      f32.inf Gn
    
    in if Gx - Gn < eps || n > max_iterations then (false, G, A, n) else

    let y_if = f32.i8 ys[i]
    let On_js = map5 (\y a d q (j, g) ->
      let b = Gx + (f32.i8 y) * g
      in if b > 0 && ((y > 0 && a > 0) || (y < 0 && a < C)) then
        let a_ = f32.max tau (D[i] + d - f32.i8 (2 * ys[i] * y) * q)
        in (-b * b / a_, j)
      else (f32.inf, -1)) ys A D Q[i] (zip iots G)

    let (_, j) = reduce_comm
      (\a b -> if b.0 <= a.0 then b else a)
      (f32.inf, -1) On_js
  
    --in if j == -1 then (false, G, A) else

    -- working set: (i, j)
    -- update alphas
    let y_jf = f32.i8 ys[j]

    let b = Gx + y_jf * G[j]
    let a = f32.max tau (D[i] + D[j] - f32.i8 (2 * ys[i] * ys[j]) * Q[i, j])
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
    let G = map3 (\g q_i q_j -> g + q_i * dA_i + q_j * dA_j) G Q[i] Q[j]
    let A[i] = A_i
    let A[j] = A_j

    in (true, G, A, n + 1)

  -- # support vectors
  let is_sv = map (\a -> i32.bool (a != 0)) A
  let n_sv = i32.sum is_sv

  let A = map2 (\a y -> a * f32.i8 y) A ys

  let coefs = filter (\a -> a != 0) A

  -- Todo: Find rho
  in (coefs, n)
