let segmented_scan 't [n] (op: t -> t -> t) (ne: t)
    (fs: [n]bool) (vs: [n]t): [n]t =
  unzip (scan (\(f1, v1) (f2, v2) ->
    (f1 || f2, if f2 then v2 else v1 `op` v2))
  (false, ne) (zip fs vs)) |> (.1)

let segmented_iota [n] (fs: [n]bool): [n]i32 =
  let is = segmented_scan (+) 0 fs (replicate n 1)
  in map (\i -> i - 1) is

-- | Distribute a list of f32's `vs` over their respective 
-- areas as indicated by the flags `fs`.
let distribute [n][m] (fs: [n]bool) (vs: [m]f32): [n]f32 =
  let is = unzip (filter (.0) (zip fs (iota n))) |> (.1) :> [m]i32
  let nd = scatter (replicate n 0) is vs
  in segmented_scan (+) 0 fs nd

let i32_distribute [n][m] (fs: [n]bool) (vs: [m]i32): [n]i32 =
  let is = unzip (filter (.0) (zip fs (iota n))) |> (.1) :> [m]i32
  let nd = scatter (replicate n 0) is vs
  in segmented_scan (+) 0 fs nd


let distribute_ends [n] (fs: [n]bool) (vs: [n]f32): [n]f32 =
  let segment_ends = rotate 1 fs
  let masked_ends = map2 (\f v -> if f then v else 0) fs vs
  let reversed_result = segmented_scan (+) 0 (reverse fs) (reverse masked_ends)
  in reverse reversed_result

let segmented_reduce 't [n] (op: t -> t -> t) (ne: t)
                            (flags: [n]bool) (as: [n]t) =
  -- Compute segmented scan.  Then we just have to fish out the end of
  -- each segment.
  let as' = segmented_scan op ne flags as
  -- Find the segment ends.
  let segment_ends = rotate 1 flags
  -- Find the offset for each segment end.
  let segment_end_offsets = segment_ends |> map i32.bool |> scan (+) 0
  let num_segments = if n > 0 then last segment_end_offsets else 0
  -- Make room for the final result.  The specific value we write here
  -- does not matter; they will all be overwritten by the segment
  -- ends.
  let scratch = replicate num_segments ne
  -- Compute where to write each element of as'.  Only segment ends
  -- are written.
  let index i f = if f then i-1 else -1
  in scatter scratch (map2 index segment_end_offsets segment_ends) as'

let f32_distribute_endings [n] (fs: [n]bool) (vs: [n]f32): [n]f32 =
  let ends = rotate 1 fs
  let masked = map2 (\f v -> f32.bool f * v) ends vs
  in (segmented_scan (+) 0 ends[::-1] masked[::-1])[::-1]

let i32_distribute_endings [n] (fs: [n]bool) (vs: [n]i32): [n]i32 =
  let ends = rotate 1 fs
  let masked = map2 (\f v -> i32.bool f * v) ends vs
  in (segmented_scan (+) 0 ends[::-1] masked[::-1])[::-1]

let i32_extract_endings [n] (n_segments: i32) (fs: [n]bool) (vs: [n]i32) =
  let ends = rotate 1 fs
  let end_offsets = ends |> map i32.bool |> scan (+) 0
  let scratch = replicate n_segments 0
  let index i f = if f then i - 1 else -1
  in scatter scratch (map2 index end_offsets ends) vs

let f32_extract_endings [n] (n_segments: i32) (fs: [n]bool) (vs: [n]f32) =
  let ends = rotate 1 fs
  let end_offsets = ends |> map i32.bool |> scan (+) 0
  let scratch = replicate n_segments 0
  let index i f = if f then i - 1 else -1
  in scatter scratch (map2 index end_offsets ends) vs