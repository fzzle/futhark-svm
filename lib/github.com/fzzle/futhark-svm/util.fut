import "../../diku-dk/segmented/segmented"

let clamp (l: f32) (x: f32) (u: f32): f32 =
  f32.max l (f32.min x u)

-- | Returns array of upper triangular indices of a n×n matrix.
-- Result has same order as if traversed by for i<n (for j=i+1..<n).
-- https://stackoverflow.com/a/27088560/2984068
let triu (n: i32): *[](i32, i32) =
  let u = n*(n-1)
  let p = u/2
  -- Converts linear index k to triu.
  let k2triu (k: i32) =
    let i = n-2-i32.f32 (f32.sqrt (f32.i32 (-8*k+4*u-7))/2-0.5)
    in (i, k+i+1-p+(n-i)*(n-i-1)/2)
  in map k2triu (iota p)

-- | Like replicate but segmented.
let segmented_replicate 't [n] (ns: [n]i32) (vs: [n]t): []t =
  map (\i -> vs[i]) (replicated_iota ns)

-- | Segmented replicate with predetermined size.
let segmented_replicate_to 't [n] (k: i32) (ns: [n]i32) (vs: [n]t): [k]t =
  segmented_replicate ns vs :> [k]t

let gather [n][m] 't (xs: [n]t) (is: [m]i32): *[m]t =
  map (\i -> xs[i]) is

-- | Exclusive prefix scan.
let exclusive_scan [n] 't (op: t -> t -> t) (ne: t) (vs: [n]t) =
  let mask i v = if bool.i32 i then v else ne
  let vs' = map2 mask (iota n) (rotate (-1) vs)
  in scan op ne vs'

-- | Counts positive numbers < k into k bins.
let bincount [n] (k: i32) (vs: [n]i32): [k]i32 =
  let bins = replicate k 0
  let ones = replicate n 1
  in reduce_by_index bins (+) 0 vs ones

let is_even (x: i32): bool =
  bool.i32 (x & 1)