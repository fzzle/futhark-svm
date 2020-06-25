local module S = import "../../diku-dk/segmented/segmented"

let clamp (l: f32) (x: f32) (u: f32): f32 =
  f32.max l (f32.min x u)

let segmented_replicate 't [n] (reps: [n]i32) (vs: [n]t): []t =
  map (\i -> vs[i]) (S.replicated_iota reps)

let exclusive_scan [n] 't (op: t -> t -> t) (ne: t) (vs: [n]t) =
  let mask i v = if bool.i32 i then v else ne
  let vs' = map2 mask (iota n) (rotate (-1) vs)
  in scan op ne vs'

let bincount [n] (k: i32) (is: [n]i32): [k]i32 =
  let bins = replicate k 0
  let ones = replicate n 1
  in reduce_by_index bins (+) 0 is ones

-- let triu_indices (n: i32): [](i32, i32) =
--   --let tot = n * (n - 1) / 2
--   map (\k ->
--     let i = n - 2 - i32.f32 (f32.sqrt (-8 * f32.i32 k + 4 * f32.i32 n *(f32.i32 n - 1) - 7)/2 - 0.5)
--     let j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2
--     in (i, j)
--   ) (iota tot)

let triu_indices (n: i32): [](i32, i32) =
  loop is = [] for i < n do
    loop is = is for j in i + 1..<n do
      is ++ [(i, j)]