-- Entrypoints for calling with Python.

import "svm"
import "types"

module svm = fsvm f32
module svc = svm.svc
module kernels = svm.kernels

module svc0 = svc kernels.linear
module svc1 = svc kernels.sigmoid
module svc2 = svc kernels.polynomial
module svc3 = svc kernels.rbf

type k_t = (f32, f32, f32)

let make_k_p0 (_: k_t) = {}
let make_k_p1 (p: k_t) = {gamma=p.0, coef0=p.1}
let make_k_p2 (p: k_t) = {gamma=p.0, coef0=p.1, degree=p.2}
let make_k_p3 (p: k_t) = {gamma=p.0}

let svc_fit [n][m] 't fit (make_k_p: k_t -> t)
    (X: [n][m]f32) (Y: [n]i32) (C: f32) (n_ws: i32)
    (max_t: i32) (max_t_in: i32) (max_t_out: i32)
    (eps: f32) (gamma: f32) (coef0: f32) (degree: f32) =
  let m_p = {n_ws, max_t, max_t_in, max_t_out, eps}
  let k_p = make_k_p (gamma, coef0, degree)
  -- Unpack for conversion to non-opaque output.
  let {weights={A,I,S,Z,R,n_c},details={O,T,T_out}}=fit X Y C m_p k_p
  in (A, I, S, Z, R, n_c, O, T, T_out)

let svc_predict [n][m][o][p][q] 't predict (make_k_p: k_t -> t)
    (X: [n][m]f32) (A: [o]f32) (I: [o]i32) (S: [p][m]f32)
    (Z: [q]i32) (R: [q]f32) (n_c: i32) (n_ws: i32)
    (gamma: f32) (coef0: f32) (degree: f32): [n]i32 =
  let m_w = {A, I, S, Z, R, n_c}
  let p_p = {n_ws}
  let k_p = make_k_p (gamma, coef0, degree)
  in predict X m_w p_p k_p

-- | Entrypoints for SVC fit.
entry svc_linear_fit     = svc_fit svc0.fit make_k_p0
entry svc_sigmoid_fit    = svc_fit svc1.fit make_k_p1
entry svc_polynomial_fit = svc_fit svc2.fit make_k_p2
entry svc_rbf_fit        = svc_fit svc3.fit make_k_p3

-- | Entrypoints for SVC predict.
entry svc_linear_predict     = svc_predict svc0.predict make_k_p0
entry svc_sigmoid_predict    = svc_predict svc1.predict make_k_p1
entry svc_polynomial_predict = svc_predict svc2.predict make_k_p2
entry svc_rbf_predict        = svc_predict svc3.predict make_k_p3
