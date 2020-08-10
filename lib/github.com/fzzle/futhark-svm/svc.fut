import "../../diku-dk/sorts/radix_sort"

import "util"
import "types"
import "kernel"
import "solver"

module svc (R: float) (S: kernel with t = R.t) = {
  local open solver R S

  -- | Train a model on X/Y.
  let fit [n][m] (X: [n][m]t) (Y: [n]i32) (C: t)
      (m_p: m_t) (k_p: s): output t [m] =
    -- Number of distinct classes.
    let n_c = 1 + i32.maximum Y
    let counts = bincount n_c Y
    let starts = exclusive_scan (+) 0 counts
    -- Sort samples by class (no negative integers).
    let sort_by_fst = radix_sort_by_key (.0) i32.num_bits i32.get_bit
    let X = map (.1) (sort_by_fst (zip Y X))
    -- Number of models to train.
    let n_m = n_c * (n_c - 1) / 2
    -- Allocate for output of trained models.
    let out = replicate n_m (0, R.i32 0, R.i32 0, 0, 0)
    let (A_I, k) = ([], 0)
    let (A_I', out', _) =
      loop (A_I, out, k) for i < n_c do
        loop (A_I, out, k) for j in i+1..<n_c do
          let (s_i, c_i) = (starts[i], counts[i])
          let (s_j, c_j) = (starts[j], counts[j])
          let n_s = c_i + c_j
          let X_i = X[s_i:s_i + c_i]
          let X_j = X[s_j:s_j + c_j]
          let X_k = concat_to n_s X_i X_j
          -- Set Y[t < c_i] = 1 and Y[t >= c_i] = -1.
          let (I_k, Y_k) = unzip (map (\t ->
            if t < c_i
            then (s_i + t, R.i32 1)
            else (s_j + t - c_i, R.i32 (-1))) (iota n_s))
          -- Solve for X_k and Y_k.
          let (A_k, o, r, t, t_out) = solve X_k Y_k (C, C) m_p k_p
          let A_I_k = filter (\x -> R.(x.0 != i32 0)) (zip A_k I_k)
          let out[k] = (length A_I_k, o, r, t, t_out)
          in (A_I ++ A_I_k, out, k + 1)
    let (A, I) = unzip A_I'
    let (Z, O, R_, T, T_out) = unzip5 out'
    -- Remove the samples from X that aren't used as support vectors.
    -- We do this by finding B_s[i] which is 1 if X[i] is used as a
    -- support vector and 0 if not.
    let trues = replicate_for I true
    let B_s = scatter (replicate n false) I trues
    let S_ = gather X (filter (\i -> B_s[i]) (iota n))
    -- Remap indices for support vectors to those of S.
    let remap = scan (+) 0 (map i32.bool B_s)
    let I = map (\i -> remap[i] - 1) I
    let weights = {A, I, S=S_, Z, R=R_, n_c}
    let details = {O, T, T_out}
    in {weights, details}

  -- | Linear kernel module for rbf computation.
  local module L = linear R
  -- | Prediction settings type.
  type p_s = prediction_settings t

  -- | Predict classes of samples X.
  let predict [n][m][o][q] (X: [n][m]t)
      ({A, I, S=S_, R=R_, Z, n_c}: weights t [m])
      ({n_ws}: prediction_settings t) (k_p: s): [n]i32 =
    let D_l_S = L.diag {} S_
    let trius = triu n_c :> [q](i32, i32)
    let F = segmented_replicate_to o Z (iota q)
    let (Y, i) = ([], 0)
    let (Y, _) = loop (Y, i) while i < n do
      let to = i32.min n (i + n_ws)
      let D_l_X = L.diag {} X[i:to]
      let K = S.matrix k_p X[i:to] S_ D_l_X D_l_S
      let Y_i = map (\K_i ->
        let dK_i = map (\j -> K_i[j]) I
        let prods = map2 (R.*) dK_i A
        -- Find decision values (without bias).
        let ds = reduce_by_index (replicate q (R.i32 0)) (R.+) (R.i32 0) F prods
        let decisions = map3 (\d r (i, j) ->
          if R.(d > r) then i else j) ds R_ trius
        let votes = bincount n_c decisions
        let max_by_fst a b = if a.0 > b.0 then a else b
        let v_c = reduce max_by_fst (0, -1) (zip votes (iota n_c))
        in v_c.1) K
      in (Y ++ Y_i, i + n_ws)
    in Y:> [n]i32
}