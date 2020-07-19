import "../util"
import "../kernels"

-- | The initial step is slightly simpler to solve. Doesn't provide a
-- huge boost here, but it does improve the chunk-based solver.
local let init [n] (K: [n][n]f32) (D: [n]f32) (P: [n]bool)
    (Y: [n]f32) (Cp: f32) (Cn: f32): ([n]f32, [n]f32) =
  let K_u = K[0]
  let V_l_I = map4 (\p d k_u i ->
    if !p -- b < 0 iff. !p since b=f_u-f_l, f_u=-1, f_l=1
    then (2 / (D[0] + d - 2 * k_u), i)
    else (-f32.inf, -1)) P D K_u (iota n)
  let max_by_fst a b = if a.0 > b.0 then a else b
  let (beta, l) = reduce_comm max_by_fst (-f32.inf, -1) V_l_I
  -- We bound d_l by both Cn and Cp as it's unlikely for 2 / eta to
  -- be greater at the initial step. Otherwise, if a_u was greater
  -- than Cp we wouldn't be able to eliminate d_u * k_u.
  -- b=2, y_u=1, y_l=-1
  let a = f32.min beta (f32.min Cn Cp) -- a_l
  let F = map3 (\y k_u k_l -> a * (k_u - k_l) - y) Y K_u K[l]
  let A = map (\i -> if i == 0 || i == l then a else 0) (iota n)
  in (F, A)

-- | Performs a single optimization step.
local let solve_step [n] (K: [n][n]f32) (D: [n]f32)
    (P: [n]bool) (Y: [n]f32) (F: [n]f32) (A: [n]f32)
    (Cp: f32) (Cn: f32) (eps: f32): (bool, f32, [n]f32, [n]f32) =
  -- Find the extreme sample x_u in X_upper, which has the minimum
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
  let y_u = Y[u]
  let y_l = Y[l]
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
  let A' = map2 (\a i ->
    if i == u then a_u else if i == l then a_l else a) A (iota n)
  in (true, d, F', A')

-- Find the objective value.
local let find_obj [n] (A: [n]f32) (F: [n]f32) (Y: [n]f32): f32 =
  -0.5 * f32.sum (map3 (\a f y -> a * (1 - f * y)) A F Y)

-- Find the bias.
local let find_rho [n] (A: [n]f32) (F: [n]f32)
    (P: [n]bool) (Cp: f32) (Cn: f32) (d: f32): f32 =
  -- Find free alphas to find rho.
  let is_free p a = a > 0 && ((p && a < Cp) || (!p && a < Cn))
  let B_f = map2 is_free P A
  let F_f = map2 (\b f -> if b then f else 0) B_f F
  let n_f = i32.sum (map i32.bool B_f)
  -- Average value of f for free alphas.
  let v_f = f32.sum F_f / f32.i32 n_f
  in if n_f > 0 then v_f else d / 2

let solve [n][m] (X: [n][m]f32) (Y: [n]f32)
    (p: parameters) (Cp: f32) (Cn: f32)
    (eps: f32) (max_it: i32) =
  -- Find full kernel matrix.
  let K = kernel_matrix p X X
  -- Cache the kernel diagonal.
  let D = kernel_diag p K
  let P = map (>0) Y
  -- Initialize A & F.
  let (F, A) = init K D P Y Cp Cn
  let (_, i, d, F, A) =
    loop (c, i, _, F, A) = (true, 1, 0, F, A) while c do
      let (b, d, F', A') = solve_step K D P Y F A Cp Cn eps
      in (b && i < max_it - 1, i + 1, d, F', A')
  let o = find_obj A F Y
  let r = find_rho A F P Cp Cn d
  -- Multiply y on alphas for prediction.
  let A = map2 (*) A Y
  -- Returns alphas, objective value, bias, and iterations.
  in (A, o, r, i)
