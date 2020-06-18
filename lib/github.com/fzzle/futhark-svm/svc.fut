import "../../diku-dk/segmented/segmented"
import "../../diku-dk/sorts/radix_sort"
import "helpers"
import "kernels"

-- Perform a single optimization step.
local let solve_step [n] (K: [n][n]f32) (D: [n]f32)
    (P: [n]bool) (Y: [n]i8) (Cp: f32) (Cn: f32)
    (F: [n]f32) (A: *[n]f32): (bool, f32, [n]f32, *[n]f32) =
  -- Find the extreme sample x_u, which has the minimum
  -- optimality indicator, f_u.
  let is_upper p a = (p && a < Cp) || (!p && a > 0)
  let B_u = map2 is_upper P A
  let F_u_I = map3 (\b f i ->
    if b then (f, i) else (f32.inf, -1)) B_u F (iota n)
  let min_by_fst a b = if a.0 < b.0 then a else b
  let (f_u, u) = reduce_comm min_by_fst (f32.inf, -1) F_u_I
  -- Find f_max so we can check if we're done.
  let is_lower p a = (p && a > 0) || (!p && a < Cn)
  let B_l = map2 is_lower P A
  let F_l = map2 (\b f -> if b then f else -f32.inf) B_l F
  let d = f32.maximum F_l - f_u
  -- Check if done.
  in if d < eps then (false, d, F, A) else
  -- Find the extreme sample x_l.
  let K_u = K[u]
  let V_l_I = map4 (\f d k_u i ->
    let b = f_u - f
    -- If u is -1 then f_u = f32.inf and b = f32.inf.
    -- If f is not in F_lower then f = -f32.inf and b = f32.inf.
    -- Checks if u isn't -1 and f in F_lower and b < 0.
    in if b < 0
       then (b * b / (D[u] + d - 2 * k_u), i)
       else (-f32.inf, -1)) F_l D K_u (iota n)
  let max_by_fst a b = if a.0 > b.0 then a else b
  let (_, l) = reduce_comm max_by_fst (-f32.inf, -1) V_l_I
  let y_u = f32.i8 Y[u]
  let y_l = f32.i8 Y[l]
  let b = f_u - F[l]
  -- eta always > 0.
  let eta = D[u] + D[l] - 2 * K_u[l]
  -- Find new a_u and a_l.
  let a_l = clamp 0 (A[l] + y_l * b / eta) Cn
  let a_u = clamp 0 (A[u] + y_l * y_u * (A[l] - a_l)) Cp
  -- Find the differences * y.
  let d_u = (a_u - A[u]) * y_u
  let d_l = (a_l - A[l]) * y_l
  -- Update optimality indicators.
  let F' = map3 (\f k_u k_l ->
    -- f + d_u * k_u + d_l * k_l
    f32.mad d_u k_u (f32.mad d_l k_l f)) F K_u K[l]
  -- Write back updated alphas.
  let A[u] = a_u
  let A[l] = a_l
  in (true, 0, F', A)

-- Find the objective value.
local let find_obj [n] (A: [n]f32) (F: [n]f32) (Y: [n]i8): f32 =
  0.5 * -f32.sum (map3 (\a f y -> a * (1 - f * f32.i8 y)) A F Y)

-- Find the bias.
local let find_rho [n] (A: [n]f32) (F: [n]f32)
    (P: [n]bool) (Cp: f32) (Cn: f32) (d: f32): f32 =
  -- Find free x to find rho.
  let is_free p a = a > 0 && ((p && a < Cp) || (!p && a < Cn))
  let B_f = map2 is_free P A
  let F_f = map2 (\b f -> if b then f else 0) B_f F
  let n_f = i32.sum (map i32.bool B_f)
  -- Average value of f for free x.
  let v_f = f32.sum F_f / f32.i32 n_f
  in if n_f > 0 then v_f else d / 2

local let solve [n][m] (X: [n][m]f32) (Y: [n]i8)
    (kernel: u8) (Cp: f32) (Cn: f32) (gamma: f32)
    (coef0: f32) (degree: f32) =
  -- Find kernel matrix / diagonal
  let (K, D) = match kernel
    case 0 -> linear X
    case 1 -> rbf X gamma
    case _ -> polynomial X gamma coef0 degree
  let A = replicate n 0
  let F = map (\y -> f32.i8 (-y)) Y
  let P = map (>0) Y
  let (_, i, d, F, A) =
    loop (c, i, _, F, A) = (true, 0, 0, F, A) while c do
      let (b, d, F, A) = solve_step K D P Y Cp Cn F A
      in (b && i < 100000000, i + 1, d, F, A)
  let o = find_obj A F Y
  let r = find_rho A F P Cp Cn d
  -- Multiply y on alphas for prediction.
  let A' = map2 (*) A (map f32.i8 Y)
  in (A', o, r, i)

-- Requires y to be 0, 1, 2...
entry train [n][m] (X: [n][m]f32) (Y: [n]u8)
    (kernel: u8) (C: f32) (gamma: f32) (coef0: f32)
    (degree: f32) =
  let sorter = radix_sort_by_key (.1) u8.num_bits u8.get_bit
  let (X, Y) = unzip (sorter (zip X Y))
  -- Number of classes.
  let k = 1 + i32.u8 (u8.maximum Y)
  let counts = bincount k (map i32.u8 Y)
  let starts = exclusive_scan (+) 0 counts
  let n_models = (k * (k - 1)) / 2
  let out = replicate n_models (0, 0, 0)
  let (A, ids, fs, out, _) = loop (A, ids, fs, out, p) = ([], [], [], out, 0) for i < k do
    let si = starts[i]
    let ci = counts[i]
    let X_i = X[si:si + ci]
    in loop (A, ids, fs, out, p) = (A, ids, fs, out, p) for j in i + 1..<k do
      let sj = starts[j]
      let cj = counts[j]
      let size = ci + cj
      let X_j = X[sj:sj + cj]
      let X_p = X_i ++ X_j :> [size][m]f32
      let Y_p = map (\x -> if x < ci then 1 else -1) (iota size)
      let is = map (+si) (iota ci)
      let js = map (+sj) (iota cj)
      let idxs = is ++ js :> [size]i32
      let (A_p, obj, rho, i) = solve X_p Y_p kernel C C gamma coef0 degree
      let (A_p, idxs) = unzip (filter (\x -> x.0 > eps) (zip A_p X_p))
      let flgs = map (==0) (iota (length idxs))
      in (A ++ A_p,
          ids ++ idxs,
          fs ++ flgs,
          out with [p] = (obj, rho, i), p + 1)

  let (objs, rhos, iter) = unzip3 out
  --let svs = bincount n ids
  --let (_, S_is) = unzip (filter (\x -> x.0 > 0) (zip svs (iota n)))
  --let n_sv = length S_is
  --let S = replicate n_sv (replicate m 0)
  --let S = loop S = S for i < n_sv do S with [i] = X[S_is[i]]
  in (A, ids, fs, rhos, objs, iter)

entry predict [n][m][o][v][s] (X: [n][m]f32) (S: [o][m]f32)
    (A: [v]f32) (rhos: [s]f32) (flags: [v]bool)
    (kernel: u8) (C: f32) (gamma: f32) (coef0: f32)
    (degree: f32) (k: i32) =
  let K = match kernel
    case _ -> map (\a -> map (\b -> dot a b) S) X
  let is = (loop is = [] for i < k do
            loop is = is for j in i + 1..<k do
              is ++ [(i, j)]) :> [s](i32, i32)
  in map (\K_i ->
    let V_f = map2 (\j a -> a * K_i[j]) (iota v) A
    let V = segmented_reduce (+) 0 flags V_f :> [s]f32
    let sgn = map2 (\v rho -> v + rho) V rhos
    let cs = map2 (\(i, j) s -> if s > 0 then i else j) is sgn
    let votes = bincount k cs
    let best = reduce (\a b -> if a.0 > b.0 then a else b)
      (i32.lowest, -1) (zip votes (iota k))
    in best.1) K