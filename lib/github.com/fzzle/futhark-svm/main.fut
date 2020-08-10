import "v2/svm"
import "v2/model"

module svm = fsvm f32
module svc = svm.svc
module kernels = svm.kernels

module k0 = svc kernels.linear
module k1 = svc kernels.sigmoid
module k2 = svc kernels.polynomial
module k3 = svc kernels.rbf

-- | Float type.
type t = f32.t

-- | Entrypoint for SVC fit.
entry svc_fit [n][m] (X: [n][m]f32) (Y: [n]i32)
    (k_id: i32) (C: f32) (n_ws: i32) (max_t: i32)
    (max_t_in: i32) (max_t_out: i32) (eps: f32)
    (gamma: f32) (coef0: f32) (degree: f32) =
  let m_p = {n_ws, max_t, max_t_in, max_t_out, eps}
  let {weights={A, I, S, Z, R, n_c}, details={O, T, T_out}} =
    match k_id
    case 0 -> k0.fit X Y C m_p {}
    case 1 -> k1.fit X Y C m_p {gamma, coef0}
    case 2 -> k2.fit X Y C m_p {gamma, coef0, degree}
    case _ -> k3.fit X Y C m_p {gamma}
  in (A, I, S, Z, R, n_c, O, T, T_out)

-- | Entrypoint for SVC predict.
entry svc_predict [n][m][o][p][q] (X: [n][m]f32)
    (k_id: i32) (n_ws: i32) (A: [o]f32) (I: [o]i32)
    (S: [p][m]f32) (R: [q]f32) (Z: [q]i32) (n_c: i32)
    (gamma: f32) (coef0: f32) (degree: f32): [n]i32 =
  let p_p = {n_ws}
  let m_w = {A, I, S, R, Z, n_c} in
    match k_id
    case 0 -> k0.predict X m_w p_p {}
    case 1 -> k1.predict X m_w p_p {gamma, coef0}
    case 2 -> k2.predict X m_w p_p {gamma, coef0, degree}
    case _ -> k3.predict X m_w p_p {gamma}
