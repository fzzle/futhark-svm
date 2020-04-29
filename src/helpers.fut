type parameters = {
  C: f32,
  gamma: f32,
  kernel: u8
}

let dot [n] (as: [n]f32) (bs: [n]f32): f32 =
  f32.sum (map2 (*) as bs)

let clamp (a: f32) (n: f32) (x: f32): f32 =
  if a < n then n else if a > x then x else a
