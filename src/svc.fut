
type parameters = {
  C: f64,
  gamma: f64,
  epsilon: f64
}

type dataset [n][m] = {
  xs: [n][m]f64,
  ys: [n]f64
}

type model = {
  p: parameters
}

let dot_prod [n] (as: [n]f64) (bs: [n]f64): f64 =
  let prods = map2 (*) as bs
  in f64.sum prods

let dot [n] (as: [n]f64) (bs: [n]f64): f64 =
  f64.sum (map2 (*) as bs)

let f [n][m] (x: [m]f64) (alphas: [n]f64) (ys: [n]f64) (xs: [n][m]f64) (b: f64): f64 =
  f64.sum (map3 (\ a y x' -> a * y * (dot x x') + b) alphas ys xs)

let lcg (seed: u64): u64 =
  (1664525u64 * seed + 1013904223u64) % 0xFFFFFFFF

let randExcept (i: i32) (lim: i32) (prev: u64): (i32, u64) =
  let p = loop prev = lcg prev while (i32.u64 prev) % lim == i do lcg prev
  in (i32.u64 p % lim, p)

let clamp (x: f64) (min: f64) (max: f64): f64 =
  if x < min then min else if x > max then max else x

let clam2 (x: f64) (min: f64) (max: f64): f64 =
  if x < min then min else f64.max x max


entry train_binary [n][m] (xs: [n][m]f64) (ys: [n]f64): ([n]f64, f64) =
  --let xs = d.xs
  --let ys = d.ys
  let r = 2132u64 -- seed
  -- todo: precompute kernel matrix
  --kernel_matrix = matrix

  -- Simplified SMO --
  -- parameters:
  let p_C = 10
  let p_tol = 0.001
  let p_max_passes = 100
  -- internal:
  let as = replicate n 0f64
  let (as, b, _, _) = loop (as, b, is, r) = (as, 0, 0, r) while is < p_max_passes do
    -- todo: Kernel in fs
    let (as, b, ca, r) = loop (as, b, ca, r) = (as, b, 0, r) for i < n do
      let x_i = xs[i]
      let y_i = ys[i]
      let a_i = as[i]
      let E_i = (f xs[i] as ys xs b) - ys[i]
      in if (ys[i] * E_i < -p_tol && as[i] < p_C) || (ys[i] * E_i > p_tol && as[i] > 0) then
        let (j, r) = randExcept i n r
        let E_j = (f xs[j] as ys xs b) - ys[j]
        let x_j = xs[j]
        let y_j = ys[j]
        let a_j = as[j]
        let a_j_old = as[j]
        let a_i_old = as[i]
        let (L, H) = if y_i != y_j then (f64.max 0 (a_j - a_i), f64.min p_C (p_C + a_j - a_i))
                                   else (f64.max 0 (a_i + a_j - p_C), f64.min p_C (a_i + a_j))
        in if L == H then (as, b, ca, r) else
          -- eta uses kernel function instead of dotprod too
          let eta = 2 * (dot_prod x_i x_j) - (dot_prod x_i x_i) - (dot_prod x_j x_j)
          in if eta >= 0 then (as, b, ca, r) else
            let a_j = a_j - (y_j * (E_i - E_j)) / eta
            let a_j = clamp a_j L H
            let as = as with [j] = a_j
            in if f64.abs(a_j - a_j_old) < 0.00001f64 then (as, b, ca, r) else
              let a_i = a_i + y_i * y_j * (a_j_old - a_j)
              let b1 = b - E_i - y_i * (a_i - a_i_old) * (dot_prod x_i x_i) - y_j * (a_j - a_j_old) * dot_prod x_i x_j
              let b2 = b - E_j - y_i * (a_i - a_i_old) * (dot_prod x_i x_j) - y_j * (a_j - a_j_old) * dot_prod x_j x_j
              let b' = if 0 < a_i && a_i < p_C then b1
                 else if 0 < a_j && a_j < p_C then b2
                 else (b1 + b2) / 2
              in (as with [i] = a_i, b', ca + 1, r)
      else (as, b, ca, r)
    in if ca == 0i32 then (as, b, is + 1i32, r)
                  else (as, b, 0, r)
  let ws = map2 (*) as ys
  in (ws, b)

--let main [n] (xs: f64[m][n]) (ys: f64[n]): ([n]f64, f64) =


let predict [n][m] (xs: [n][m]f64): [n]i16 =
  -- predict: model -> xs -> ys
  replicate n 0i16
