let segmented_scan 't [n] (op: t -> t -> t) (ne: t)
    (fs: [n]bool) (vs: [n]t): [n]t =
  unzip (scan (\(f1, v1) (f2, v2) ->
    (f1 || f2, if f2 then v2 else op v1 v2))
  (false, ne) (zip fs vs)) |> (.1)

let segmented_iota [n] (fs: [n]bool): [n]i32 ->
  let is = segmented_scan (+) 0 fs (replicate n 1)
  in map (\i -> i - 1) is

-- | Distribute a list of f32's `vs` over their respective 
-- areas as indicated by the flags `fs`.
let distribute [n][m] (fs: [n]bool) (vs: [m]f32): [n]f32 =
  let is = unzip (filter (.0) (zip fs (iota n))) |> (.1) :> [m]i32
  let nd = scatter (replicate n 0) is vs
  in segmented_scan fs (+) 0 nd

let distribute_ends [n] (fs: [n]bool) (vs: [n]f32): [n]f32 =
  let segment_ends = rotate 1 fs
  let masked_ends = map2 (\f v -> if f then v else 0) fs vs
  let reversed_result = segmented_scan (+) 0 (reverse fs) (reverse masked_ends)
  in reverse reversed_result