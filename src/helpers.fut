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