import "../../diku-dk/segmented/segmented"

type parameters = {
  C: f32,
  gamma: f32,
  kernel: u8
}

let eps: f32 = 1e-3
let tau: f32 = 1e-12

let dot [n] (as: [n]f32) (bs: [n]f32): f32 =
  f32.sum (map2 (*) as bs)

let clamp (a: f32) (n: f32) (x: f32): f32 =
  f32.min x (f32.max n a)

-- | Distribute a list of f32's `vs` over their respective 
-- areas as indicated by the flags `fs`.
entry distribute [n][m] (fs: [n]bool) (vs: [m]f32): [n]f32 =
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