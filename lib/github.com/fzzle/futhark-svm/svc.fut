import "../../diku-dk/sorts/radix_sort"
import "helpers"

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
  -- Find the extreme example x_l.
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
  let a_l = clamp (A[l] + y_l * b / eta) 0 Cn
  let a_u = clamp (A[u] + y_l * y_u * (A[l] - a_l)) 0 Cp
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
local let find_obj [n] (A: [n]f32) (F: [n]f32) (Y: [n]i8) =
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
    (Cp: f32) (Cn: f32) (kernel: f32 -> f32):
    ([]f32, f32, f32, i32) =
  let max_iter = 10000000
  -- Find kernel / kernel diagonal
  let K = map (\x0 -> map (\x1 -> kernel (dot x0 x1)) X) X
  let D = map (\i -> K[i, i]) (iota n)

  let A = replicate n 0
  let F = map (\y -> f32.i8 (-y)) Y
  let P = map (>0) Y
  -- Solve.
  let (_, i, d, F, A) = loop (c, i, _, F, A) = (true, 0, 0, F, A) while c do
    let (b, d, F, A) = solve_step K D P Y Cp Cn F A
    in (b && i < max_iter, i + 1, d, F, A)

  let obj = find_obj A F Y
  let rho = find_rho A F P Cp Cn d
  let A = map2 (*) A (map f32.i8 Y)
  let A = filter (\a -> f32.abs a > eps) A

  in (A, obj, rho, i)

local let bincount [n] (k: i32) (is: [n]i32): [k]i32 =
  let bins = replicate k 0
  let ones = replicate n 1
  in reduce_by_index bins (+) 0 is ones

-- Requires y to be 0, 1, 2...
entry train [n][m] (X: [n][m]f32) (Y: [n]u8)
    (kernel: u8) (C: f32) (gamma: f32) (coef0: f32)
    (degree: f32) =
  let sorter = radix_sort_by_key (.1) u8.num_bits u8.get_bit
  let (X, Y) = unzip (sorter (zip X Y))
  let l = 1 + i32.u8 (u8.maximum Y)
  let counts = bincount l (map i32.u8 Y)
  let starts = scan (+) 0 (rotate (-1) counts) |> map (+(-last counts))
  -- k(k-1)/2 models
  let n_models = (l * (l - 1)) / 2
  let objs = replicate n_models 0
  let rhos = replicate n_models 0
  let iter = replicate n_models 0
  let (A, objs, rhos, iter, _) = loop (A, objs, rhos, iter, k) = ([], objs, rhos, iter, 0) for i in 0..<l do
    let X_i = X[starts[i]:starts[i] + counts[i]]
    in loop (A, objs, rhos, iter, k) = (A, objs, rhos, iter, k) for j in i + 1..<l do
      let size = counts[i] + counts[j]
      let X_j = X[starts[j]:starts[j] + counts[j]]
      let X_k = X_i ++ X_j :> [size][m]f32
      let Y_k = map (\x -> if x < counts[i] then 1 else -1) (iota size)
      let solver = solve X_k Y_k C C
      let (A_k, obj, rho, i) = match kernel
        case 1 -> solver (\x -> (gamma * x + coef0) ** degree)
        case _ -> solver id
      in (A ++ A_k,
          objs with [k] = obj,
          rhos with [k] = rho,
          iter with [k] = i, k + 1)

  --let (A, I) = unzip A
  --let SV = map (\i -> X[i]) I
  in (A, objs, rhos, iter)

-- X: Samples to be predicted.
-- S: Support vectors.
-- entry predict [n][m][k][v] (X: [n][m]f32) (S: [k][m]f32)
--     (A: [v]f32) (rhos: []f32) (flags: [v]bool) =
--   let K = map (\x -> map (\s -> dot x s) S) X
--   map (\i ->
--     let K_i = K[i] -- X[i]'s row
--     let V_f = map2 (\j -> a * K_i[j]) (iota v) A
--     let V_s = segmented_reduce (+) 0 flags V_f
--     ) (iota n)