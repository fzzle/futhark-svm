import "../../../diku-dk/sorts/radix_sort"
import "../util"
import "../kernels"

-- Perform a single optimization step.
local let solve_step [n][m] (K: [n][m]f32) (D: [m]f32)
    (P: [m]bool) (Y: [n]i8) (F: [m]f32) (I: [m]i32) (A: *[m]f32)
    (Cp: f32) (Cn: f32) (eps: f32): (bool, f32, [m]f32, *[m]f32) =
  -- Find the extreme sample x_u in X_upper, which has the minimum
  -- optimality indicator, f_u.
  let is_upper p a = (p && a < Cp) || (!p && a > 0)
  let B_u = map2 is_upper P A
  let F_u_I = map3 (\b f i ->
    if b then (f, i) else (f32.inf, -1)) B_u F (iota m)
  let min_by_fst a b = if a.0 < b.0 then a else b
  -- Can use reduce_comm because order doesn't matter at all.
  let (f_u, u) = reduce_comm min_by_fst (f32.inf, -1) F_u_I
  -- Find f_max so we can check if we're done.
  let is_lower p a = (p && a > 0) || (!p && a < Cn)
  let B_l = map2 is_lower P A
  let F_l = map2 (\b f -> if b then f else -f32.inf) B_l F
  let d = f32.maximum F_l - f_u
  -- Check if done.
  in if d < eps then (false, d, F, A) else
  -- Find the extreme sample x_l.
  let i_u = I[u]
  let K_u = K[i_u]
  let V_l_I = map4 (\f d k_u i ->
    let b = f_u - f
    -- If u is -1 then f_u = f32.inf and b = f32.inf.
    -- If f is not in F_lower then f = -f32.inf and b = f32.inf.
    -- Checks if u isn't -1 and f in F_lower and b < 0.
    in if b < 0
       then (b * b / (D[u] + d - 2 * k_u), i)
       else (-f32.inf, -1)) F_l D K_u (iota m)
  let max_by_fst a b = if a.0 > b.0 then a else b
  let (_, l) = reduce_comm max_by_fst (-f32.inf, -1) V_l_I
  let i_l = I[l]
  let y_u = f32.i8 Y[i_u]
  let y_l = f32.i8 Y[i_l]
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
    f32.mad d_u k_u (f32.mad d_l k_l f)) F K_u K[I[l]]
  -- Write back updated alphas.
  let A[u] = a_u
  let A[l] = a_l
  in (true, d, F', A)

-- Find the objective value.
local let find_obj [n] (A: [n]f32) (F: [n]f32) (Y: [n]i8): f32 =
  -0.5 * f32.sum (map3 (\a f y -> a * (1 - f * f32.i8 y)) A F Y)

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

let ws = 1024
let half_ws = ws / 2

let solve [n][m] (X: [n][m]f32) (Y: [n]i8)
    (k: kernel) (Cp: f32) (Cn: f32) (gamma: f32)
    (coef0: f32) (degree: f32) (eps: f32) (max_iter: i32) =
  let sorter = radix_sort_float_by_key (.0) f32.num_bits f32.get_bit
  let max_outer_iter = 100
  let max_inner_iter = 100000

  -- Initialize A / F.
  let A = replicate n 0
  let F = map (\y -> f32.i8 (-y)) Y
  let P = map (>0) Y
  let (_, i, t, d, _, F, A) =
    loop (c, i, t, d0, d_data, F, A) = (true, 0, 0, 0, (0, 0, 0, 0), F, A) while c do
      let F_I = sorter (zip F (iota n))
      let is_upper p a = (p && a < Cp) || (!p && a > 0)
      let is_lower p a = (p && a > 0) || (!p && a < Cn)
      let B_ul = map2 (\p a -> (is_upper p a, is_lower p a)) P A
      let B_ul = map (\(_, i) -> B_ul[i]) F_I
      let (B_u, B_l) = unzip B_ul

      --let B_u = map (\(_, i) -> is_upper P[i] A[i]) F_I
      let scan_u = scan (+) 0 (map i32.bool B_u)
      let scan_l = scan (+) 0 (map i32.bool B_l)
      let n_u = i32.min (last scan_u) half_ws
      let n_l = (last scan_l) - n_u
      let U_s = map2 (&&) B_u (map (<=n_u) scan_u)
      let L_s = map2 (&&) B_l (map (>n_l) scan_l)
      let s = map2 (||) U_s L_s
      --let B_l = map (\(_, i) -> is_lower P[i] A[i]) F_I
      let (F_s, I_s) = unzip (unzip (filter (.0) (zip s F_I))).1

      let A_s = map (\i -> A[i]) I_s
      let Y_s = map (\i -> Y[i]) I_s
      let P_s = map (\i -> P[i]) I_s
      let X_s = map (\i -> X[i]) I_s
      let K_s = kernel_matrix X X_s k gamma coef0 degree
      let D_s = map2 (\i sel_i -> 1) (indices I_s) I_s

      let (b0, d0, F_s'0, A_s'0) = solve_step K_s D_s P_s Y F_s I_s A_s Cp Cn eps

      let local_eps = f32.max eps (d0 * 0.1)

      let (b, t', d, _, A_s') =
        loop (c, t, _, F_s, A_s) = (b0, 1, 0, F_s'0, A_s'0) while c do
          let (b, d, F_s', A_s') = solve_step K_s D_s P_s Y F_s I_s A_s Cp Cn local_eps
          in (b && t < max_inner_iter, t + 1, d, F_s', A_s')

      let A_s = map (\i -> A[i]) I_s
      let diff_s = map3 (\a' a y -> (a' - a) * f32.i8 y) A_s' A_s Y_s
      let F' = map2 (\f K_f -> f + f32.sum (map2 (*) K_f diff_s)) F K_s
      let A' = scatter A I_s A_s'

      let (d0_prev, d0_prev_prev, sm_lo_d, sm_sh_d) = d_data

      let sm_lo_d = if f32.abs (d0 - d0_prev) < eps * 0.001 then sm_lo_d + 1 else 0
      let sm_sh_d = if f32.abs (d0 - d0_prev_prev) < eps * 0.001 then sm_sh_d + 1 else 0
      let stop = d0 < eps || (sm_lo_d >= 10 && f32.abs(d0 - 2) > eps) || (sm_sh_d >= 10 && f32.abs(d0 - 2) > eps)

      in (!stop && i != max_outer_iter, i + 1, t + t', d0, (d0, d0_prev, sm_lo_d, sm_sh_d), F', A')
  let o = find_obj A F Y
  let r = find_rho A F P Cp Cn d
  -- Multiply y on alphas for prediction.
  let A = map2 (*) A (map f32.i8 Y)
  -- Returns alphas, indices of support vectors,
  -- objective value, bias, and iterations.
  in (A, o, r, i)
