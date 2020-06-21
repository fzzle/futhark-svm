import "../../diku-dk/sorts/radix_sort"
import "util"
import "kernels"

-- Perform a single optimization step.
local let solve_step [n] (K: [n][n]f32) (D: [n]f32)
    (P: [n]bool) (Y: [n]i8) (F: [n]f32) (A: *[n]f32)
    (Cp: f32) (Cn: f32) (eps: f32): (bool, f32, [n]f32, *[n]f32) =
  -- Find the extreme sample x_u, which has the minimum
  -- optimality indicator, f_u.
  let is_upper p a = (p && a < Cp) || (!p && a > 0)
  let B_u = map2 is_upper P A
  let F_u_I = map3 (\b f i ->
    if b then (f, i) else (f32.inf, -1)) B_u F (iota n)
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
    (k: kernel) (Cp: f32) (Cn: f32) (gamma: f32)
    (coef0: f32) (degree: f32) (eps: f32) (max_iter: i32) =
  -- Find full kernel matrix.
  let K = kernel_matrix X X k gamma coef0 degree
  -- Cache the kernel diagonal.
  let D = kernel_diag K k
  -- Initialize A / F.
  let A = replicate n 0
  let F = map (\y -> f32.i8 (-y)) Y
  let P = map (>0) Y
  let (_, i, d, F, A) =
    loop (c, i, _, F, A) = (true, 0, 0, F, A) while c do
      let (b, d, F, A) = solve_step K D P Y F A Cp Cn eps
      in (b && i < max_iter, i + 1, d, F, A)
  let o = find_obj A F Y
  let r = find_rho A F P Cp Cn d
  -- Multiply y on alphas for prediction.
  let A = map2 (*) A (map f32.i8 Y)
  -- Returns alphas, indices of support vectors,
  -- objective value, bias, and iterations.
  in (A, o, r, i)

-- Requires y to be 0, 1, 2...
entry train [n][m] (X: [n][m]f32) (Y: [n]u8) (k_id: i32)
    (C: f32) (gamma: f32) (coef0: f32) (degree: f32)
    (eps: f32) (max_iter: i32) =
  let sorter = radix_sort_by_key (.1) u8.num_bits u8.get_bit
  let (X, Y) = unzip (sorter (zip X Y))
  let k = kernel_from_id k_id
  -- t: Number of classes.
  let t = 1 + i32.u8 (u8.maximum Y)
  let counts = bincount t (map i32.u8 Y)
  let starts = exclusive_scan (+) 0 counts
  --let n_models = (t * (t - 1)) / 2
  --let out = replicate n_models (0, 0, 0)
  let (A, I, F, out, _) =
    loop (A, I, F, out, p) = ([], [], [], [], 0) for i < t do
    let si = starts[i]
    let ci = counts[i]
    let X_i = X[si:si + ci]
    in loop (A, I, F, out, p) = (A, I, F, out, p) for j in i + 1..<t do
      let sj = starts[j]
      let cj = counts[j]
      let size = ci + cj
      let X_j = X[sj:sj + cj]
      let X_p = concat_to size X_i X_j
      let Y_p = map (\x -> if x < ci then 1 else -1) (iota size)
      let I_p = concat_to size (map (+si) (iota ci)) (map (+sj) (iota cj))
      let (A_p, obj, rho, i) = solve X_p Y_p k C C gamma coef0 degree eps max_iter
      -- Only keep non-zero alphas.
      let (A_p, I_p) = unzip (filter ((.0) >-> (!=0)) (zip A_p I_p))
      let flgs = replicate (length A_p) p
      in (A ++ A_p,
          I ++ I_p,
          F ++ flgs,
          out ++ [(obj, rho, i)], p + 1)

  let (O, R, iter) = unzip3 out

  --tmp
  let li = length I
  let I = I :> [li]i32

  -- Remove the samples from X that aren't used as support vectors.
  -- We do this by finding S_b[i] which is 1 if X[i] is used as a
  -- support vector and 0 if not.
  let bins = replicate n false
  let trues = replicate li true
  let B_s = reduce_by_index bins (||) false I trues
  let (_, S) = unzip (filter (.0) (zip B_s X))
  -- Remap indices for support vectors.
  let remap = scan (+) 0 (map i32.bool B_s)
  let I = map (\i -> remap[i] - 1) I
  in (A, I, S, F, R, O, iter, t)


let segmented_reduce [n] 't (op: t -> t -> t) (ne: t)
    (flags: [n]bool) (as: [n]t) (n_segments: i32) =
  let is = map (\x -> x - 1) (scan (+) 0 (map i32.bool flags))
  let scratch = replicate n_segments ne
  in reduce_by_index scratch op ne is as


entry predict [n][m][o][v][s] (X: [n][m]f32) (S: [o][m]f32)
    (A: [v]f32) (I: [v]i32) (rhos: [s]f32) (F: [v]i32)
    (t: i32) (k_id: i32) (gamma: f32) (coef0: f32) (degree: f32)
    (ws: i32) =
  let k = kernel_from_id k_id
  let is = triu_indices t :> [s](i32, i32)
  let (p, _) = loop (p, i) = ([], 0) while i < n do
    let to = i32.min (i + ws) n
    let K = kernel_matrix X[i:to] S k gamma coef0 degree
    in (p ++ map (\K_i ->
      let dK_i = map (\j -> K_i[j]) I
      let prods = map2 (*) dK_i A
      let sums = reduce_by_index (replicate s 0) (+) 0 F prods
      let classes = map3 (\s rho (i, j) -> if s > rho then i else j) sums rhos is
      let votes = bincount t classes
      let max_by_fst a b = if a.0 > b.0 then a else b
      in (reduce max_by_fst (i32.lowest, -1) (zip votes (iota t))).1) K, i + ws)
  in p
