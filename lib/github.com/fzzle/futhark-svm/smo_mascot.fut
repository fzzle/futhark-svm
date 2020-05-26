import "helpers"

entry solve [n][m] (xs: [n][m]f32) (ys: [n]i8): ([]f32, f32, i32, i32) =
  let C = 10
  let max_iterations = 100000

  -- Q[i, j] = y[i] * y[j] * K[i, j]
  let K = map (\x -> map (\x' -> dot x x') xs) xs
  let D = map (\i -> K[i, i]) (iota n)

  let A = replicate n 0f32
  let ysf32 = map f32.i8 ys
  let G = map (\y -> -y) ysf32
  let ysbool = map (\y -> y > 0) ys
  let iots = iota n

  let (_, k, G, A) = loop (c, k, G, A) = (true, 0, G, A) while c do

    let f_upper_is = map4 (\y a g i -> if (y && a < C) || (!y && a > 0) then (g, i) else (f32.inf, -1)) ysbool A G iots
    let f_lower = map3 (\y a g -> if (y && a > 0) || (!y && a < C) then g else -f32.inf) ysbool A G
    let (f_u, u) = reduce_comm (\a b -> if a.0 < b.0 then a else b) (f32.inf, -1) f_upper_is
    let f_max = f32.maximum f_lower
    
    in if f_u >= f_max || k > max_iterations then (false, k, G, A) else

    let tmp_is = map4 (\f_i d_i K_u_i i ->
      let d = f_u - f_i
      in if d < 0 then -- also checks if f_i is X_lower, since f_i = -inf if in X_lower.
        let eta = D[u] + d_i - 2 * K_u_i[u]
        in ((d * d) / eta, i)
      else (-f32.inf, -1)) 
      f_lower D K iots

    let (d2_eta, l) = reduce_comm
      (\a b -> if a.0 >= b.0 then a else b)
      (-f32.inf, -1) tmp_is

    -- working set: (i, j)
    -- update alphas

    let f_l = G[l]
    
    let d_a_u = if ysbool[u] then C - A[u] else A[u]
    let d_a_l = f32.min (if ysbool[l] then A[l] else C - A[n]) (d2_eta / (f_u - f_l))
    let delta = f32.min d_a_u d_a_l 
      --f32.minimum [
      --  (if ysbool[u] then C - A[u] else A[u])
      --  (if ysbool[l] then A[l] else C - A[l])
      --  (d2_eta / (f_u - f_l))
      --]

    -- update gradient
    let G = map3 (\g q_u q_l -> g - delta * (q_l - q_u)) G K[u] K[l]
    let A[u] = A[u] + delta * ysf32[u]
    let A[l] = A[l] - delta * ysf32[l]

    in (true, k + 1, G, A)

  -- # support vectors
  let obj = (reduce (+) 0 (map2 (*) A (map (\g -> g-1) G)))/2

  --let A = map2 (\a y -> a * f32.i8 y) A ys

  let coefs = filter (\a -> a > eps || a < -eps) A

  -- Todo: Find rho
  in (coefs, obj, k, length coefs)
