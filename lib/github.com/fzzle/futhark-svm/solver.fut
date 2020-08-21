import "../../diku-dk/sorts/radix_sort"

import "util"
import "sets"
import "types"
import "kernel"

module solver (R: float) (S: kernel with t = R.t) = {
  -- Imports is_upper, is_lower, is_free.
  local open sets R

  -- | Kernel parameters type.
  type s = S.s
  -- | Float type.
  type t = R.t
  -- | Model settings type.
  type m_t = training_settings R.t
  -- | Weighted C pair type.
  type C_t = (t, t)

  -- | Lower bound on eta.
  local let tau = R.f32 1e-12

  -- | Performs a single optimization step. Returns a bool indicating
  -- if it has reached the threshold model.eps, a pair of f_u and f_l,
  -- and the updated F, A.
  local let solve_step [n] (K: [n][n]t) (D: [n]t)
      (Y: [n]t) (F: [n]t) (A: [n]t) ((Cp, Cn): C_t)
      (m_p: m_t): (bool, (t, t), [n]t, [n]t) =
    -- Find the extreme sample x_u in X_upper, which has the minimum
    -- optimality indicator, f_u.
    let B_u = map2 (is_upper Cp) Y A
    let F_u_I = map3 (\b f i ->
      if b then (f, i) else (R.inf, -1)) B_u F (iota n)
    let min_by_fst a b = if R.(a.0 < b.0) then a else b
    let (f_u, u) = reduce min_by_fst (R.inf, -1) F_u_I
    -- Find f_max so we can check if we're done.
    let B_l = map2 (is_lower Cn) Y A
    let F_l = map2 (\b f -> if b then f else R.(negate inf)) B_l F
    let f_max = R.maximum F_l
    -- Check if done.
    in if R.(f_max-f_u<m_p.eps) then (false, (f_u, f_max), F, A) else
    -- Find the extreme sample x_l.
    let K_u = K[u]
    let V_l_I = map4 (\f d k_u i ->
      let b = R.(f_u - f)
      -- If u is -1 then f_u = f32.inf and b = f32.inf.
      -- If f is not in F_lower then f = -f32.inf and b = f32.inf.
      -- Checks if u isn't -1 and f in F_lower and b < 0.
      in if R.(b < i32 0)
        then (R.(b * b / (max tau (D[u] + d - i32 2 * k_u))), i)
        else (R.(negate inf), -1)) F_l D K_u (iota n)
    let max_by_fst a b = if R.(a.0 > b.0) then a else b
    let (_, l) = reduce max_by_fst (R.(negate inf), -1) V_l_I
    -- Find bounds for a_u, a_l.
    let c_u = R.(if Y[u] > i32 0 then Cp - A[u] else A[u])
    let c_l = R.(if Y[l] < i32 0 then Cn - A[l] else A[l])
    let eta = R.(max tau (D[u] + D[l] - i32 2 * K_u[l]))
    let b = R.(F[l] - f_u)
    -- Find new a_u and a_l.
    let q = R.(min (min c_u c_l) (b / eta))
    -- Find the differences * y.
    let a_u = R.(A[u] + q * Y[u])
    let a_l = R.(A[l] - q * Y[l])
    -- Update optimality indicators.
    let F' = map3 (\f k_u k_l -> R.(f + q * (k_u - k_l))) F K_u K[l]
    -- Write back updated alphas.
    let A' = map2 (\a i ->
      if i == u then a_u else if i == l then a_l else a) A (iota n)
    -- q should always be greater than 0. Under unusual circumstances
    -- (such as floating-point underflow) where q is 0, we signal
    -- that the solver is finished, since it'll be stuck otherwise.
    in (R.(q > i32 0), (f_u, f_max), F', A')

  -- | Computes the objective value.
  local let find_obj [n] (Y: [n]t) (F: [n]t) (A: [n]t): t =
    R.(f32 (-0.5) * sum (map3 (\y f a -> a * (i32 1 - f * y)) Y F A))

  -- | Computes the bias.
  local let find_rho [n] (Y: [n]t) (F: [n]t) (A: [n]t)
      ((Cp, Cn): C_t) (v:t): t =
    -- Find free alphas to find rho.
    let B_f = map2 (is_free Cp Cn) Y A
    let F_f = map2 (\b f -> if b then f else R.i32 0) B_f F
    let n_f = i32.sum (map i32.bool B_f)
    -- Average value of f for free alphas.
    let v_f = R.(sum F_f / i32 n_f)
    in if n_f > 0 then v_f else R.(v * f32 (-0.5))

  -- | Solves the intial step given a full kernel K, initializing
  -- optimality indicators F and alphas A.
  local let init_step_pre [n] (K: [n][n]t) (D: [n]t)
      (Y: [n]t) ((Cp, Cn): C_t): ([n]t, [n]t) =
    let K_u = K[0]
    let V_l_I = map4 (\y d k_u i ->
      if R.(y < i32 0) -- b<0 iff. !p since b=f_u-f_l, f_u=-1, f_l=1
      then (R.(i32 2 / (D[0] + d - i32 2 * k_u)), i)
      else (R.(negate inf), -1)) Y D K_u (iota n)
    let max_by_fst a b = if R.(a.0 > b.0) then a else b
    let (beta, l) = reduce max_by_fst (R.(negate inf), -1) V_l_I
    -- We bound d_l by both Cn and Cp as it's unlikely for 2 / eta to
    -- be greater at the initial step. Otherwise, if a_u was greater
    -- than Cp we wouldn't be able to eliminate d_u * k_u.
    -- b=2, y_u=1, y_l=-1
    let a = R.min beta (R.min Cp Cn) -- a_l
    let F = map3 (\y k_u k_l -> R.(a * (k_u - k_l) - y)) Y K_u K[l]
    let A = map (\i -> if i==0 || i==l then a else R.i32 0) (iota n)
    in (F, A)

  -- | Linear kernel for rbf computation.
  local module L = linear R

  -- | Solves the quadratic programming problem by SMO. It computes
  -- the full kernel matrix and solves for all samples.
  let solve_full [n][m] (X: [n][m]t) (Y: [n]t) (C: C_t)
      (m_p: m_t) (k_p: s): ([n]t, t, t, i32, i32) =
    -- Compute linear diagonal (for rbf).
    let D_r = L.diag () X
    -- Compute full kernel matrix.
    let K = S.matrix k_p X X D_r D_r
    -- Cache the kernel diagonal.
    let D = S.extdiag k_p K
    -- Initialize F & A.
    let (F, A) = init_step_pre K D Y C
    -- d = (f_u, f_l) for the initial f_u=-1 and f_l=1
    let (c, i, d) = (true, 1, R.((i32 (-1), i32 1)))
    let (_, i, d, F, A) =
      loop (c, i, d, F, A) while c && i != m_p.max_t do
        let (b, d', F', A') = solve_step K D Y F A C m_p
        in (b, i + 1, d', F', A')
    let o = find_obj Y F A
    let r = find_rho Y F A C (R.(d.0+d.1))
    -- Multiply y on alphas for prediction.
    let A = map2 (R.*) A Y
    -- Returns alphas, objective value, bias, and iterations.
    in (A, o, r, i, 0)

  -- | Performs the initial step for the two-level decomposition
  -- solver. It's useful because we can select any pair of u and l
  -- (w/ y_u=1 and y_l=-1) and compute their kernel row. Thereby,
  -- it's possible to perform an initial step, and then a sorting of
  -- the optimality indicators and select a better initial working
  -- set than if we simply went by +1, -1. With u = 0.
  local let init_step_ws [n][m] (X: [n][m]t) (D: [n]t) (D_r: [n]t)
      (Y: [n]t) ((Cp, Cn): C_t) (k_p: s): (*[n]t, *[n]t) =
    let K_u = S.row k_p X X[0] D_r D_r[0]
    let V_l_I = map4 (\y d k_u i ->
      if R.(y < i32 0) -- b<0 iff. !p since b=f_u-f_l, f_u=-1, f_l=1
      then (R.(i32 2 / (D[0] + d - i32 2 * k_u)), i)
      else (R.(negate inf), -1)) Y D K_u (iota n)
    let max_by_fst a b = if R.(a.0 > b.0) then a else b
    let (beta, l) = reduce max_by_fst (R.(negate inf), -1) V_l_I
    let K_l = S.row k_p X X[l] D_r D_r[l]
    let a = R.min beta (R.min Cp Cn) -- a_l
    let F = map3 (\y k_u k_l -> R.(a * (k_u - k_l) - y)) Y K_u K_l
    let A = map (\i -> if i==0 || i==l then a else R.i32 0) (iota n)
    in (F, A)

  type cache [n_ws][n] = {
    p0: [n_ws][n]t,
    p1: [n_ws][n]t,
    -- ic: [n_ws]i32,  -- Next/current I_ws
    i0: [n_ws]i32,  -- Previous I_ws
    i1: [n_ws]i32   -- PrevPrevious I_Ws
  }

  let refer [n][m] (X: [n][m]t) (D_r: [n]t) (k_p: s) (n_ws: i32)
      (c: cache [n_ws][n]) (I_ws: [n_ws]i32) (i: i32):
      (cache [n_ws][n], [n][n_ws]t) =

    let c_p1: [n_ws][n]t = c.p1
    let c_i1: [n_ws]i32  = c.i1
    in if i < 2 then -- If not warm
      let X_ws = gather X I_ws
      let D_r_ws = gather D_r I_ws
      let K_ws_t = S.matrix k_p X_ws X D_r_ws D_r
      let cache' = {p0=K_ws_t, p1=c.p0, i0=I_ws, i1=c.i0}
      in (cache', transpose K_ws_t)
    else
    -- Find ws values that are cached.
    let I_c = #[incremental_flattening(only_intra)]
      map (\i_ws -> find_unique i_ws c_i1) I_ws
    -- Boolean vector w/ true if miss.
    let B_m = map (==(-1)) I_c
    -- I_h: Hit indices, I_m: Miss indices.
    let (I_m, I_h) = partition (\i -> B_m[i]) (iota n_ws)
    -- TODO: Use cached rows .p, computing 1024 ws colums less.
    -- Partition in working_set and save indices not in ws.

    -- map (\i -> ) cache.pi
    let I_ws_m = gather I_ws I_m
    let X_ws_m = gather X I_ws_m
    let D_r_ws_m = gather D_r I_ws_m
    let K_ws_m = S.matrix k_p X_ws_m X D_r_ws_m D_r
    -- Find indices of computed rows to put them back.
    let I_m' = map (\i -> i - 1) (scan (+) 0 (map i32.bool B_m))
    -- Assemble the rows.
    let K_ws_t = map3 (\b_m i_m i_c ->
      if b_m then K_ws_m[i_m] else c_p1[i_c]) B_m I_m' I_c

    let cache' = {p0=K_ws_t, p1=c.p0, i0=I_ws, i1=c.i0}
    -- Cache end
    in (cache', transpose K_ws_t)

  -- | Assumes that ws < n
  let working_set [n][m] (X: [n][m]t) (D_r: [n]t) (k_p: s)
      (Y: [n]t) (F: [n]t) (A: [n]t) ((Cp, Cn): C_t) (n_ws: i32)
      (p0: [n_ws][n]t) (p1: [n_ws][n]t) (i0: [n_ws]i32) (i1: [n_ws]i32)
      (i: i32): ([n_ws][n]t, [n_ws][n]t, [n_ws]i32, [n_ws]i32) = --: *[n_ws]i32 =
    -- Get indices of sorted optimality indicators F.
    let I = map (.1) (radix_sort_float R.num_bits
      (\i (f, _) -> R.get_bit i f) (zip F (iota n)))

    let n_sel = if i > 0 then n_ws/2 else n_ws
    let selected = scatter (replicate n false) i0 (replicate n_ws true)
    let B_u = map3 (\s y a -> !s && is_upper Cp y a) selected Y A
    let B_l = map3 (\s y a -> !s && is_lower Cn y a) selected Y A
    let (B_u, B_l) = unzip (gather (zip B_u B_l) I)
    let T_u = scan (+) 0 (map i32.bool B_u)
    -- n_u: Spaces allotted to upper.
    let n_u = n_sel / 2
    -- n_avail -= n_u
    let S_u = map2 (\b_u t_u -> b_u && t_u <= n_u) B_u T_u
    let B_l' = map2 (\b_l' s_u -> b_l' && !s_u) B_l S_u
    let T_l' = scan (+) 0 (map i32.bool B_l')
    -- ws - n_u: Spaces left. n_l: n lower to discard.
    let n_l = last T_l' - (n_sel / 2)
    let S_ = map3 (\s b_l' t_l' -> s || (b_l' && t_l' > n_l)) S_u B_l' T_l'
    let n_selected = reduce (+) 0 (map i32.bool S_)

    -- Check if the working set has been filled. If there are still
    -- open slots we fill them with available samples from upper.
    let n_open = n_ws - n_selected

    -- Put back at original indices.
    -- in (unzip (filter (.0) (zip S I))).1 :> *[ws]i32
    let S' = scatter (replicate n false) I S_
    let I_ws = filter (\i -> S'[i]) (iota n)
    let I_ws' = (I_ws ++ i0[0:n_open]) :> [n_ws]i32

    -- in if i < 2 then -- If not warm
    --   let X_ws = gather X I_ws
    --   let D_r_ws = gather D_r I_ws
    --   let K_ws_t = S.matrix k_p X_ws X D_r_ws D_r
    --   let K_ws_t = concat_to n_ws K_ws_t c.p0[0:n_open]
    --   let cache' = {p0=K_ws_t, p1=c.p0, i0=I_ws', i1=c.i0}
    --   in cache'
    -- else
    -- Find ws values that are cached.
    let I_c = -- #[incremental_flattening(only_intra)]
      map (\i_ws -> #[sequential] find_unique i_ws i1) I_ws
    -- Boolean vector w/ true if miss.
    let B_m = map (==(-1)) I_c
    -- I_h: Hit indices, I_m: Miss indices.
    let (I_m, I_h) = partition (\i -> B_m[i]) (iota (n_ws - n_open))
    -- TODO: Use cached rows .p, computing 1024 ws colums less.
    -- Partition in working_set and save indices not in ws.

    -- map (\i -> ) cache.pi
    let I_ws_m = gather I_ws I_m
    let X_ws_m = gather X I_ws_m
    let D_r_ws_m = gather D_r I_ws_m
    let K_ws_m = S.matrix k_p X_ws_m X D_r_ws_m D_r
    -- Find indices of computed rows to put them back.
    let I_m' = map (\i -> i - 1) (scan (+) 0 (map i32.bool B_m))
    -- Assemble the rows.
    -- let K_ws_t = map3 (\b_m i_m i_c ->
      -- if b_m then K_ws_m[i_m] else c_p1[i_c]) B_m I_m' I_c

    let I_c_h = gather I_c I_h
    let I_ws_h = gather I_ws I_h
    let K_ws_h = gather p1 I_c_h
    let I_ws' =  (I_ws_h ++ I_ws_m ++ i0[0:n_open]) :> [n_ws]i32
    let K_ws_t = (K_ws_h ++ K_ws_m ++ p0[0:n_open]) :> [n_ws][n]t

    -- Cache end
    in (K_ws_t, p0, I_ws', i0)


    -- in (I_ws ++ c.i0[0:n_open]) :> [n_ws]i32


  let d_eps: t = R.f32 1e-6

  -- | Solve the QP problem by SMO and two-level decomposition.
  let solve_ws [n][m] (X: [n][m]t) (Y: [n]t) (C: C_t)
      (m_p: m_t) (k_p: s): ([n]t, t, t, i32, i32) =
    let n_ws = m_p.n_ws
    -- Compute the kernel diagonal.
    let D = S.diag k_p X
    -- Compute linear diagonal (only for rbf).
    let D_r = S.diag k_p X
    -- Initialize F & A.
    let (F, A) = init_step_ws X D D_r Y C k_p
    -- i: Outer iterations, j: Inner.
    -- Cache
    let pK = replicate n_ws (replicate n (R.i32 0))
    let prev_I_ws = replicate n_ws (-1i32)
    let d = {d0=R.i32 2, d1=R.inf, swap=0i32, same=0i32, s0=R.i32 0}

    let p0=pK
    let p1=pK
    let i0=prev_I_ws
    let i1=prev_I_ws

    let eps_ws = R.f32 0.2

    let (c, i, j) = (true, 0, 1)
    let (_, i, j, d, _, F, A, _, _, _, _) =
      loop (c, i, j, d, eps_ws, F, A, p0, p1, i0, i1) while c && i < m_p.max_t_out do
      -- We can spare finding working set + finding K_ws if we simply
      -- check the entire dataset if we're done.
      -- let (K_ws, I_ws) = select_ws X Y C lru model k_p
      let (p0', p1', i0', i1') = working_set X D_r k_p Y F (copy A) C n_ws p0 p1 i0 i1 i
      let I_ws = i0'
      let K_ws = transpose p0'
      -- let (cache', K_ws) = refer X D_r k_p n_ws cache I_ws i
      -- Gather ws data.
      let D_ws = gather D I_ws
      let Y_ws = gather Y I_ws
      -- Get full kernel rows for ws.
      let K_wsx2 = gather K_ws I_ws
      -- F, A
      let F_ws = gather F I_ws
      let A_ws = gather A I_ws
      -- Solve the working set problem
      let (c1, k) = (true, 0)
      let (_, k, _, A_ws') = loop (c1, k, F_ws, A_ws) while c1 && k < m_p.max_t_in do
        let (b, _, F_ws', A_ws') = solve_step K_wsx2 D_ws Y_ws F_ws A_ws C m_p
        in (b, k + 1, F_ws', A_ws')
      -- Update F and write back A_ws to A.
      let d_ws = map3 (\a' a y -> R.((a' - a) * y)) A_ws' A_ws Y_ws
      let F' = map2 (\f K_i -> R.(f + sum (map2 (*) K_i d_ws))) F K_ws
      let A' = scatter A I_ws A_ws'
      -- Update difference infos.
      let B_u = map2 (is_upper C.0) Y A'
      let B_l = map2 (is_lower C.1) Y A'
      let F_u = map2 (\b f -> if b then f else R.inf) B_u F
      let F_l = map2 (\b f -> if b then f else R.(negate inf)) B_l F
      let f_u = R.minimum F_u
      let f_l = R.maximum F_l
      let s0 = R.(f_u + f_l)
      let d0 = R.(f_l - f_u)
      let same = if R.(abs (d0 - d.d0) < d_eps) then d.same + 1 else 0
      let swap = if R.(abs (d0 - d.d1) < d_eps) then d.swap + 1 else 0
      let stop = R.(f_l - f_u < m_p.eps) || same >= 10 || swap >= 10
      let d' = {d0, d1=d.d0, s0, same, swap}
      let eps_ws = R.(max m_p.eps (d0*f32 0.1))
      in (!stop, i + 1, j + k, d', eps_ws, F', A', p0', p1', i0', i1')
    let o = find_obj Y F A
    let r = find_rho Y F A C d.s0
    -- Multiply y on alphas for prediction.
    let A = map2 (R.*) Y A
    -- Returns alphas, objective value, bias, iterations,
    -- and outer iterations.
    in (A, o, r, j, i)

  -- | Applies the solver suitable to the problem. If there are fewer
  -- samples than n_ws, there's a significant memory and time overhead
  -- when using the two-level decomposition solver (solve_ws), and the
  -- full-kernel solver (solve_full) is used instead.
  let solve [n][m] (X: [n][m]t) (Y: [n]t) (C: C_t)
      (m_p: m_t) (k_p: s): ([n]t, t, t, i32, i32) =
    if n <= m_p.n_ws
    then solve_full X Y C m_p k_p
    else solve_ws   X Y C m_p k_p
}