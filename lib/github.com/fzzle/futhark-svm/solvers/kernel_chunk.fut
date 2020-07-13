import "../../../diku-dk/sorts/radix_sort"
import "../util"
import "../kernels"

let ws = 1024
let tau = 1e-6f32

-- Performs the initial step before we start sorting the optimality
-- indicators, such that it has more to go by than simply +1, -1.
local let init_step [n][m] (X: [n][m]f32) (D: [n]f32)
    (P: [n]bool) (Y: [n]f32) (Cp: f32) (Cn: f32)
    (p: parameters): ([n]f32, [n]f32) =
  let K_u = kernel_row p X X[0]
  let V_l_I = map4 (\p d k_u i ->
    if !p -- b < 0 iff. !p since b=f_u-f_l, f_u=-1, f_l=1
    then (2 / (D[0] + d - 2 * k_u), i)
    else (-f32.inf, -1)) P D K_u (iota n)
  let max_by_fst a b = if a.0 > b.0 then a else b
  let (beta, l) = reduce_comm max_by_fst (-f32.inf, -1) V_l_I
  let K_l = kernel_row p X X[l]
  -- We bound d_l by both Cn and Cp as it's unlikely for 2 / eta to
  -- be greater at the initial step. Otherwise, if a_u was greater
  -- than Cp we wouldn't be able to eliminate d_u * k_u.
  -- b=2, y_u=1, y_l=-1
  let a = f32.min beta (f32.min Cn Cp) -- a_l
  let F = map3 (\y k_u k_l -> a * (k_u - k_l) - y) Y K_u K_l
  let A = map (\i -> if i == 0 || i == l then a else 0) (iota n)
  in (F, A)

-- Perform a single optimization step.
local let solve_inner_step [n] (K: [n][n]f32) (D: [n]f32)
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
  let A' = map2 (\i a ->
    if i == u then a_u else if i == l then a_l else a) (iota n) A
  in (true, d, F', A')

-- local let solve_outer_step [n][m] =
--   let sorter = radix_sort_float_by_key (.0) f32.num_bits f32.get_bit
--   let (_, I) = unzip (sorter (zip F (iota n)))
--   let F_s

-- Find the objective value.
local let find_obj [n] (A: [n]f32) (F: [n]f32) (Y: [n]f32): f32 =
  -0.5 * f32.sum (map3 (\a f y -> a * (1 - f * y)) A F Y)

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

let get_working_set [n] (I: [n]i32) (P: [n]bool) (A: [n]f32)
    (Cp: f32) (Cn: f32) =
  let is_upper p a = (p && a < Cp) || (!p && a > 0)
  let is_lower p a = (p && a > 0) || (!p && a < Cn)
  let B_ul = map2 (\p a -> (is_upper p a, is_lower p a)) P A
  let (B_u, B_l) = unzip (map (\i -> B_ul[i]) I)
  let S_u = scan (+) 0 (map i32.bool B_u)
  let S_l = scan (+) 0 (map i32.bool B_l)
  let n_u = i32.min (last S_u) (ws / 2)
  let n_l = (last S_l) - n_u
  let s = map4 (\b_u s_u b_l s_l ->
    (b_u && s_u <= n_u) || (b_l && s_l > n_l)) B_u S_u B_l S_l
  in (unzip (filter (.0) (zip s I))).1 -- s[i]

let solve [n][m] (X: [n][m]f32) (Y: [n]f32)
    (p: parameters) (Cp: f32) (Cn: f32)
    (eps: f32) (max_iter: i32) =
  let sort = radix_sort_float_by_key (.0) f32.num_bits f32.get_bit
  let max_outer_iter = 100
  let max_inner_iter = 100000
  -- let max_iter

  let P = map (>0) Y
  let D = compute_kernel_diag p X
  let (F, A) = init_step X D P Y Cp Cn p
  let d = {p=2f32, pp=f32.inf, swap=0, same=0}
  let statics = zip3 Y P D

  let i = 0
  let j = 1
  let c = true
  let (_, i, j, d, F, A) =
    loop (c, i, j, d, F, A) while c do
      let (_, I) = unzip (sort (zip F (iota n)))
      let I_ws = get_working_set I P A Cp Cn

      let F_s = map (\i -> F[i]) I_ws
      let A_s = map (\i -> A[i]) I_ws
      let X_s = #[sequential] map (\i -> X[i]) I_ws
      let (Y_s, P_s, D_s) = unzip3 (map (\i -> statics[i]) I_ws)
      let K = kernel_matrix p X X_s
      let K_s = map (\i -> K[i]) I_ws
      -- Find the global difference f_l - f_u
      let (b0, d0, F_s0, A_s0) = solve_inner_step K_s D_s P_s Y_s F_s A_s Cp Cn eps

      -- let stop = !b0 || d.same >= 10 || d.swap >= 10
      -- in if stop then (false, i, j, )

      let local_eps = f32.max eps (d0 * 0.1)
      -- Solve the working set problem
      let (_, t', _, A_s') =
        loop (c, t, F_s, A_s) = (b0, 1, F_s0, A_s0) while c do
          let (b, _, F_s', A_s') = solve_inner_step K_s D_s P_s Y_s F_s A_s Cp Cn local_eps
          in (b && t < max_inner_iter, t + 1, F_s', A_s')

      let diff_s = map3 (\a' a y -> (a' - a) * y) A_s' A_s Y_s
      let F' = map2 (\f col -> f + f32.sum (map2 (*) diff_s col)) F K
      let A' = scatter A I_ws A_s'

      let same = if f32.abs (d0 - d.p)  < tau then d.same + 1 else 0
      let swap = if f32.abs (d0 - d.pp) < tau then d.swap + 1 else 0
      let stop = d0 < eps || d.same >= 10 || d.swap >= 10
      let d' = {p=d0, pp=d.p, same, swap}
      in (!stop && i != max_outer_iter, i + 1, j + t', d', F', A')
  let obj = find_obj A F Y
  let rho = find_rho A F P Cp Cn d.p
  -- Multiply y on alphas for prediction.
  let A = map2 (*) A Y
  -- Returns alphas, objective value, bias, and iterations.
  in (A, obj, rho, i)

