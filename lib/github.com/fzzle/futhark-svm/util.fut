import "../../diku-dk/segmented/segmented"

-- | Returns array of upper triangular indices of a n×n matrix.
-- Result has same order as if traversed by for i<n (for j=i+1..<n).
-- https://stackoverflow.com/a/27088560/2984068
let triu (n: i64): *[](i64, i64) =
  let u = n * (n - 1)
  let p = u / 2
  -- Converts linear index k to triu.
  let k2triu (k: i64) =
    let i = n - 2 - i64.f32 (f32.sqrt (f32.i64 (-8*k+4*u-7))/2-0.5)
    in (i, k + i + 1 - p + (n - i) * (n - i - 1) / 2)
  in map k2triu (0..1..<p)

-- | Like replicate but segmented.
let segmented_replicate 't [n] (ns: [n]i64) (vs: [n]t): []t =
  map (\i -> vs[i]) (replicated_iota ns)

-- | Segmented replicate with predetermined size.
let segmented_replicate_to 't [n] (k: i64) (ns: [n]i64) (vs: [n]t): [k]t =
  segmented_replicate ns vs :> [k]t

-- | Produces vs[i] for every i in is.
let gather [n][m] 't (vs: [n]t) (is: [m]i64): *[m]t =
  map (\i -> vs[i]) is

-- | Exclusive prefix scan.
let exclusive_scan [n] 't (op: t -> t -> t) (ne: t) (vs: [n]t) =
  let mask i v = if bool.i64 i then v else ne
  let vs' = map2 mask (iota n) (rotate (-1) vs)
  in scan op ne vs'

-- | Counts positive numbers < k into k bins.
let bincount [n] (k: i64) (vs: [n]i64): [k]i64 =
  let bins = replicate k 0
  let ones = replicate n 1
  in reduce_by_index bins (+) 0 vs ones

-- | Get index of an element in vs which contains the element at
-- most once. Returns -1 if not found.
let find_unique [n] (e: i64) (vs: [n]i64): i64 =
  let is = map2 (\v i -> if v == e then i else -1) vs (iota n)
  in i64.maximum is

-- | Replicate a value for each element of array.
let replicate_for 't 's [n] (_: [n]t) (v: s): *[n]s =
  replicate n v
