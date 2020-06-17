local let dot [n] (a: [n]f32) (b: [n]f32): f32 =
  f32.sum (map2 (*) a b)

-- Squared euclidean distance
local let sqdist [n] (a: [n]f32) (b: [n]f32): f32 =
  f32.sum (map (\x -> x * x) (map2 (-) a b))

let linear [n][m] (X: [n][m]f32): ([n][n]f32, [n]f32) =
  let K = map (\x -> map (dot x) X) X
  let D = map (\i -> K[i, i]) (iota n)
  in (K, D)

let polynomial [n][m] (X: [n][m]f32) (gamma: f32)
    (coef0: f32) (degree: f32): ([n][n]f32, [n]f32) =
  let f a b = (gamma * dot a b + coef0) ** degree
  let K = map (\x -> map (f x) X) X
  let D = map (\i -> K[i, i]) (iota n)
  in (K, D)

let rbf [n][m] (X: [n][m]f32) (gamma: f32): ([n][n]f32, [n]f32) =
  let f a b = f32.exp (-gamma * sqdist a b)
  let K = map (\x -> map (f x) X) X
  let D = replicate n 1
  in (K, D)