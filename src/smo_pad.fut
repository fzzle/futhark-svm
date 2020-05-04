import "helpers"
let solve [n][m] (xs: [n][m]f32) (ys: [n]i8) (k: i32): [k]f32 =
  let C = 10

  -- Q[i, j] = y[i] * y[j] * K[i, j]
  let Q = map2 (\ x y -> map2 (\ x' y' -> f32.i8 (y * y') * dot x x') xs ys) xs ys
  
  let A = replicate k 0f32
  let G = replicate n (-1f32)

  -- let s = (A, G)
  let (_, _, A) = loop (c, G, A) = (true, G, A) while c do
    -- working set selection 3
    let (i, Gx) = loop (i, Gx) = (-1, f32.lowest) for t < n do
      let y_t = ys[t]
      let A_t = A[t]
      let y_tf = f32.i8 y_t
      let Gx' = -y_tf * G[t]
      let c0 = (y_t == 1 && A_t < C) || (y_t == -1 && A_t > 0)
      let c1 = Gx' >= Gx
      in if c0 && c1 then (t, Gx') else (i, Gx)

    let y_if = f32.i8 ys[i]
    let (j, _, Gn) = loop (j, on, Gn) = (-1, f32.highest, f32.highest) for t < n do
      let y_t = ys[t]
      let A_t = A[t]
      let y_tf = f32.i8 y_t
      let Gn' = -y_tf * G[t]
      let c0 = (y_t == 1 && A_t > 0) || (y_t == -1 && A_t < C)
      let c1 = Gn' <= Gn
      in if !c0 then (j, on, Gn) else

      let b = Gx + y_tf * G[t]
      let Gn' = if c1 then Gn' else Gn
      in if b <= 0 then (j, on, Gn') else

      let a = Q[i, i] + Q[t, t] - 2f32 * y_if * y_tf * Q[i, t]
      let a = if a <= 0 then tau else a
      let on' = -(b * b) / a
      let c2 = on' <= on
      in if c2 then (t, on', Gn') else (j, on, Gn')

    let c0 = Gx - Gn < eps
    let c1 = j == -1
    in if c0 || c1 then (false, G, A) else

    let y_jf = f32.i8 ys[j]

    -- working set: (i, j)
    let a = Q[i, i] + Q[j, j] - 2 * y_if * y_jf * Q[i, j]
    let a = if a <= 0 then tau else a
    let b = (-y_if) * G[i] + y_jf * G[j]

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

entry pad [n][m] (xs: [n][m]f32) (ys: [n]u8): *[][][]f32 =
  let counts = reduce_by_index (replicate 256 0) (+) 0
    (map i32.u8 ys) (replicate n 1)
  let buckets = scan (\ a c -> a + i32.bool (c > 0)) (-1) counts
  let labels = map (.1) (filter (\ z -> z.0 > 0) (zip counts (iota 256)))
  let size = i32.maximum counts
  let classes = length labels
  let k = (size * 2)
  let As = map (\ l ->
    let d1 = unzip (filter (\ (_, y) -> y == u8.i32 l) (zip xs ys))
    in map (\ l' -> 
      let d2 = unzip (filter (\ (_, y) -> y == u8.i32 l') (zip xs ys))
      let (xs1, ys1) = d1
      let (xs2, ys2) = d2
      let ys1 = map (\_-> 1) ys1
      let ys2 = map (\_-> -1) ys2 
      let x = concat_to (counts[l] + counts[l']) xs1 xs2
      let y = concat_to (counts[l] + counts[l']) ys1 ys2
      in solve x y k
      ) labels
    ) labels
  in As

