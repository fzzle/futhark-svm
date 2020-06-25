import "../../diku-dk/sorts/radix_sort"
import "util"
import "kernels"
import "solvers/tsvm_chunk_based"

-- Requires y to be 0, 1, 2...
entry train [n][m] (X: [n][m]f32) (Y: [n]u8) (k_id: i32)
    (C: f32) (gamma: f32) (coef0: f32) (degree: f32)
    (eps: f32) (max_iter: i32) =
  let sorter = radix_sort_by_key (.0) u8.num_bits u8.get_bit
  let (Y, X) = unzip (sorter (zip Y X))
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
  -- We do this by finding B_s[i] which is 1 if X[i] is used as a
  -- support vector and 0 if not.
  let bins = replicate n false
  let trues = replicate li true
  let B_s = reduce_by_index bins (||) false I trues
  let (_, S) = unzip (filter (.0) (zip B_s X))
  -- Remap indices for support vectors.
  let remap = scan (+) 0 (map i32.bool B_s)
  let I = map (\i -> remap[i] - 1) I
  in (A, I, S, F, R, O, iter, t)

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
