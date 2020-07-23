import "../../../diku-dk/sorts/radix_sort"
import "../util"
import "../kernels"

-- TODO: If it can be done with a similar to kernel_chunk,
-- move this into kernel_chunk, and perhaps try to make it work for
-- kernel_full as well - since that can be done.

let ws_size = 1024
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
  let a = f32.min beta (f32.min Cn Cp)
  let F = map3 (\y k_u k_l -> a * (k_u - k_l) - y) Y K_u K_l
  let A = map (\i -> if i == 0 || i == l then a else 0) (iota n)
  in (F, A)

-- Perform a single optimization step.
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
  let A = map2 (\i a ->
    if i == u then a_u else if i == l then a_l else a) (iota n) A
  in (true, d, F', A)

--
let working_set_batch [n][o] (F: [o][n]f32) (A: [o][n]f32)
    (P: [n]bool) (Cp: [o]f32) (Cn: [o]f32) =
  -- let A = A[0]
  -- let F = F[0]
  -- let Cp = Cp[0]
  -- let Cn = Cn[0]
  -- let sort = radix_sort_float_by_key (.0) f32.num_bits f32.get_bit
  -- let (_, I) = unzip (sort (zip F (iota n)))
  -- let is_upper p a = (p && a < Cp) || (!p && a > 0)
  -- let is_lower p a = (p && a > 0) || (!p && a < Cn)
  -- let B_ul = map2 (\p a -> (is_upper p a, is_lower p a)) P A
  -- let (B_u, B_l) = unzip (map (\i -> B_ul[i]) I)
  -- let S_u = scan (+) 0 (map i32.bool B_u)
  -- let S_l = scan (+) 0 (map i32.bool B_l)
  -- let n_u = i32.min (last S_u) (ws_size / 2)
  -- let n_l = (last S_l) - n_u
  -- let s = map4 (\b_u s_u b_l s_l ->
  --   (b_u && s_u <= n_u) || (b_l && s_l > n_l)) B_u S_u B_l S_l
  -- in (unzip (filter (.0) (zip s I))).1

  let sort = radix_sort_float_by_key (.0) f32.num_bits f32.get_bit
  let (is, scores) = unzip (map4 (\F A Cp Cn ->
    let I = map (.1) (sort (zip F (iota n)))
    let is_upper p a = (p && a < Cp) || (!p && a > 0)
    let is_lower p a = (p && a > 0) || (!p && a < Cn)
    let B_ul = map2 (\p a -> (is_upper p a, is_lower p a)) P A
    let (B_u, B_l) = unzip (map (\i -> B_ul[i]) I)
    let S_u = scan (+) 0 (map i32.bool B_u)
    let S_l = scan (+) 0 (map i32.bool B_l)
    let n_u = i32.min (last S_u) (ws_size / 2)
    let n_l = (last S_l) - n_u
    let score = map4 (\b_u s_u b_l s_l ->
      if s_u == 1 && b_u then f32.inf else
      if s_l == (last S_l) && b_l then f32.inf else
      -- TODO: Base score on how extreme the indicator is.
      if (b_u && s_u <= n_u) || (b_l && s_l > n_l) then 1 else 0) B_u S_u B_l S_l
    in (I, score)
  ) F A Cp Cn)
  let size = n*o
  let is = flatten is :> [size]i32
  let scores = flatten scores :> [size]f32
  let seggrescores = reduce_by_index (replicate n 0) (+) 0 is scores
  in (unzip (sort (zip seggrescores (iota n)))[n - ws_size:n]).1

-- | Find the objective value.
local let find_obj [n] (A: [n]f32) (F: [n]f32) (Y: [n]f32): f32 =
  -0.5 * f32.sum (map3 (\a f y -> a * (1 - f * y)) A F Y)

-- | Find the bias.
local let find_rho [n] (A: [n]f32) (F: [n]f32)
    (P: [n]bool) (Cp: f32) (Cn: f32) (d: f32): f32 =
  -- Find free x to find rho.
  let is_free p a = a > 0 && ((p && a < Cp) || (!p && a < Cn))
  let B_f = map2 is_free P A
  let F_f = map2 (\b f -> if b then f else 0) B_f F
  let n_f = i32.sum (map i32.bool B_f)
  -- Average value of f for free x.
  let v_f = f32.sum F_f / f32.i32 n_f
  in if n_f > 0 then v_f else d / 2 --f

local let apply_kernel (p: parameters) (x: f32) =
  match p.kernel
  case #rbf -> assert false x -- not implemented yet
  case #linear -> x
  case #polynomial -> (p.gamma * x + p.coef0) ** p.degree
  case #sigmoid -> f32.tanh (p.gamma * x + p.coef0)



let solve_batch_seq [n][m][o] (X: [n][m]f32) (Y: [n]f32)
    (ps: [o]parameters) (Cps: [o]f32) (Cns: [o]f32)
    (eps: f32) (max_iter: i32) =
  -- TODO: If ws < o * 2 then either set ws = o*2 or fail since it
  -- might not solve it correctly otherwise, as global extreme
  -- indicators need to be part of the ws.

  let max_outer_iter = 100
  let max_inner_iter = 100000

  let P = map (>0) Y

  -- todo: don't compute kernel diag
  let Dt = map (\p_t -> compute_kernel_diag p_t X) ps
  let (F, A) = unzip (map4 (\D_t Cp_t Cn_t p_t -> init_step X D_t P Y Cp_t Cn_t p_t) Dt Cps Cns ps)
  let ds = replicate o {p=f32.inf, pp=f32.inf, swap=0, same=0}

  let i = 0
  let t = 0
  let c = true
  let As_c = A
  let Fs_c = F


  let (_, i, t, Fs_c, As_c, ds) =
    loop (c, i, t, Fs_c, As_c, ds) while c do
      let I_ws = working_set_batch Fs_c As_c P Cps Cns
      let Y_ws = map (\i -> Y[i]) I_ws
      let P_ws = map (\i -> P[i]) I_ws
      let X_ws = map (\i -> X[i]) I_ws
      let p_linear = {kernel=#linear, gamma=0, coef0=0, degree=0}
      let K_lin = kernel_matrix p_linear X X_ws

      let As_c_t = transpose As_c
      -- let As_c_ws = map (\A -> gather A I_ws) As_c
      let As_c_ws = transpose (map (\i -> As_c_t[i]) I_ws)


      let (Fs_c', As_c_ws', ds') = loop (Fs_c, As_c_ws, ds) for i < o do
        let p = ps[i]
        let d = ds[i]
        let Cp = Cps[i]
        let Cn = Cns[i]

        let K_ws = map (\a -> map (\b -> apply_kernel p b) a) K_lin
        let K_wsx2 = gather K_ws I_ws
        let D_ws = kernel_diag p K_wsx2
        let solve_step' = solve_step K_wsx2 D_ws P_ws Y_ws
        -- F, A
        let F_ws = gather Fs_c[i] I_ws
        let A_ws = gather As_c[i] I_ws
        -- Perform one step to find the global difference d0=f_l-f_u.
        let (b0, d0, F_ws0, A_ws0) = solve_step' F_ws A_ws Cp Cn eps
        -- Check if we're done: If d0 < eps or if it's stuck
        -- (using the same heuristics as fsvm).
        let same = if f32.abs (d0 - d.p)  < tau then d.same + 1 else 0
        let swap = if f32.abs (d0 - d.pp) < tau then d.swap + 1 else 0
        let stop = !b0 || same >= 10 || swap >= 10
        -- Update difference infos.
        let d' = {p=d0, pp=d.p, same, swap}
        -- TODO: Find way to remove stopped
        let ds[i] = d'
        in if stop then (Fs_c, As_c_ws, ds) else
        let eps_ws = f32.max eps (d0 * 0.1)
        -- Solve the working set problem
        let (c1, k) = (true, 1)
        let (_, k, _, A_ws') = loop (c1, k, F_ws, A_ws) = (c1, k, F_ws0, A_ws0) while c1 do
          let (b, _, F_ws', A_ws') = solve_step' F_ws A_ws Cp Cn eps_ws
          in (b && k < max_inner_iter, k + 1, F_ws', A_ws')

        -- Update F and write back A_ws to A.
        let d_ws = map3 (\a' a y -> (a' - a) * y) A_ws' A_ws Y_ws
        let Fs_c[i] = map2 (\f K_i -> f + f32.sum (map2 (*) d_ws K_i)) Fs_c[i] K_ws
        let As_c_ws[i] = A_ws'--scatter (copy As_c[i]) I_ws A_ws'
        -- let As_c_ws[i] = A_ws'
        in (Fs_c, As_c_ws, ds)

      let As_c' = transpose (scatter As_c_t I_ws (transpose As_c_ws'))
      -- let As_c' = map (\A A_ws' -> scatter A A_ws' I_ws) As_c As_c_ws'

      let stop = all (\d -> d.p < eps) ds'
      in (!stop && i != max_outer_iter, i + 1, 0, Fs_c', As_c', ds')
  -- Multiply y on alphas for prediction.
  let As = map (\A -> map2 (*) A Y) As_c
  -- Returns alphas, objective value, bias, and iterations.
  let os = map2 (\A F -> find_obj A F Y) As_c Fs_c
  let rs = map5 (\A F Cp Cn d -> find_rho A F P Cp Cn d.p) As_c Fs_c Cps Cns ds --replicate o 0
  let is = replicate o i
  in (As, os, rs, is)

