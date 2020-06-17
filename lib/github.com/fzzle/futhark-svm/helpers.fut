import "../../diku-dk/segmented/segmented"

let eps: f32 = 1e-3
let tau: f32 = 1e-12

let dot [n] (as: [n]f32) (bs: [n]f32): f32 =
  f32.sum (map2 (*) as bs)

let clamp (l: f32) (x: f32) (u: f32): f32 =
  f32.max l (f32.min x u)

let mean [n] (xs: [n]f32): f32 =
  (f32.sum xs) / (f32.i32 n)

let segmented_reduce' [n] 't (op: t -> t -> t) (ne: t)
    (n_segments: i32) (flags: [n]bool) (segment_end_indices: [n]i32)
    (as: [n]t) =
  let as' = segmented_scan op ne flags as
  in scatter (replicate n_segments ne) segment_end_indices as'

let segmented_replicate 't [n] (reps: [n]i32) (vs: [n]t): []t =
  map (\i -> vs[i]) (replicated_iota reps)

let exclusive_scan [n] 't (op: t -> t -> t) (ne: t) (vs: [n]t) =
  let mask i v = if bool.i32 i then v else ne
  let vs' = map2 mask (iota n) (rotate (-1) vs)
  in scan op ne vs'

let bincount [n] (k: i32) (is: [n]i32): [k]i32 =
  let bins = replicate k 0
  let ones = replicate n 1
  in reduce_by_index bins (+) 0 is ones
