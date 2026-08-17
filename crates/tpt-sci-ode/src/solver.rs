//! From-scratch ODE integration for `tpt-sci-ode`.
//!
//! Four methods are provided, all implemented in-house on top of the dual-
//! licensed `tpt-math` dense linear algebra (no `diffsol`/`nalgebra`/`faer` in
//! the shipped graph):
//!
//! * [`Method::Tsit45`] — explicit Runge–Kutta (Tsitouras 4(5)), non-stiff.
//! * [`Method::TrBdf2`] — 2-stage SDIRK (TR-BDF2), A-stable, stiff.
//! * [`Method::Esdirk34`] — 4-stage ESDIRK order 3(4), A-/L-stable, stiff.
//! * [`Method::Bdf`] — variable-order (1–5) backward differentiation, stiff.
//!
//! A shared adaptive-step driver (`integrate`) handles step-size control and
//! dense output via Hermite interpolation.

use crate::error::OdeError;
use crate::linalg::{eval, jacobian, norm2, solve_newton_system, RhsFn};
use crate::problem::OdeProblem;

/// Integration method selection for [`OdeProblem::solve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Backward Differentiation Formulae — variable-order, stiff-capable,
    /// handles singular mass matrices. The default for general problems.
    Bdf,
    /// Explicit (non-stiff) Runge–Kutta (Tsitouras 4(5)). Cheap when the system
    /// is non-stiff.
    Tsit45,
    /// Trapezoidal-rule BDF2 (SDIRK/ESDIRK family) — stiff-capable, A-stable.
    TrBdf2,
    /// Explicit-first-stage singly-diagonally-implicit RK of order 3(4) — stiff-
    /// capable, A-/L-stable.
    Esdirk34,
}

/// Result of a single successful step.
struct StepResult {
    /// New time.
    t: f64,
    /// New state.
    y: Vec<f64>,
    /// Per-component local error estimate (at the new state).
    err: Vec<f64>,
    /// Derivative at the start of the step (for Hermite interpolation).

    /// Derivative at the end of the step.
    f_new: Vec<f64>,
}

/// Attempt one step of the given method; returns `Ok` on success with the new
/// state and a local error estimate, or `Err` if the nonlinear solve could not
/// converge (caller should shrink `h` and retry). `bdf_state`, if `Some`, is the
/// BDF order/history; it is updated in place on success.
fn try_step(
    method: Method,
    f: &RhsFn,
    t: f64,
    y: &[f64],
    h: f64,
    rtol: f64,
    atol: f64,
    bdf_state: Option<&mut BdfState>,
) -> Result<StepResult, OdeError> {
    match method {
        Method::Tsit45 => step_tsit45(f, t, y, h, rtol, atol),
        Method::TrBdf2 => step_trbdf2(f, t, y, h, rtol, atol),
        Method::Esdirk34 => step_esdirk34(f, t, y, h, rtol, atol),
        Method::Bdf => step_bdf(f, t, y, h, rtol, atol, bdf_state),
    }
}

/// Adaptive-step driver shared by `solve` and `solve_dense`.
fn integrate(
    prob: &OdeProblem,
    method: Method,
    t_final: f64,
    t_eval: Option<&[f64]>,
) -> Result<Vec<Vec<f64>>, OdeError> {
    let f = &*prob.rhs;
    let rtol = prob.rtol;
    let atol = prob.atol;

    let mut t = prob.t0;
    let mut y = prob.y0.clone();
    let mut f_cur = eval(f, t, &y);
    let dir = if t_final >= t { 1.0 } else { -1.0 };
    let span = (t_final - t).abs();

    let h_min = 1e-14_f64.max(span * 1e-12);
    let h_max = span; // never step past the whole interval at once
    let max_steps = 100_000usize;
    let safety = 0.9_f64;

    let mut h = (span * 1e-3).max(1e-6).min(h_max);
    if dir < 0.0 {
        h = -h;
    }

    let mut bdf_state = if method == Method::Bdf {
        let mut st = BdfState::new();
        // Seed the history with the initial point so the first (order-1) BDF
        // corrector has a valid y_{n-1} (backward Euler bootstrap).
        st.push(prob.t0, prob.y0.clone(), f_cur.clone());
        Some(st)
    } else {
        None
    };

    // Output bookkeeping.
    let mut outputs: Vec<Vec<f64>> = Vec::new();
    let mut eval_idx = 0usize;
    if let Some(te) = t_eval {
        for &tev in te {
            if dir * (tev - t) <= 0.0 {
                return Err(OdeError::invalid(
                    "t_eval entries must be strictly increasing and beyond t0",
                ));
            }
        }
    }

    let mut steps = 0usize;
    loop {
        let h_to_final = dir * (t_final - t);
        if h_to_final <= 0.0 {
            break;
        }
        if h.abs() > h_to_final {
            h = dir * h_to_final;
        }
        if let Some(te) = t_eval {
            if eval_idx < te.len() {
                let h_to_eval = dir * (te[eval_idx] - t);
                if h.abs() > h_to_eval {
                    h = dir * h_to_eval;
                }
            }
        }
        if h.abs() < h_min {
            return Err(OdeError::StepTooSmall { t });
        }

        match try_step(method, f, t, &y, h, rtol, atol, bdf_state.as_mut()) {
            Ok(res) => {
                let err_est = weighted_norm(&res.err, &res.y, rtol, atol);
                let accept = err_est <= 1.0 || h.abs() <= h_min * 2.0;
                if !accept {
                    // Reject: shrink and retry without advancing.
                    let mut next = h * safety * 0.5_f64.max(err_est.powf(-0.2));
                    if dir < 0.0 {
                        next = -next.abs();
                    }
                    if next.abs() < h_min {
                        return Err(OdeError::StepTooSmall { t });
                    }
                    h = next;
                    continue;
                }

                // Accept: grow step for next time, record outputs, advance.
                let mut next = h * safety * (if err_est < 1e-12 {
                    2.0
                } else {
                    err_est.powf(-0.2).min(5.0)
                });
                if dir < 0.0 {
                    next = -next.abs();
                }
                next = dir * next.abs().min(h_max);

                record_outputs(
                    t, &y, &f_cur, &res, t_eval, &mut eval_idx, dir, &mut outputs,
                );

                // Update BDF history on accept.
                if let Some(st) = bdf_state.as_mut() {
                    st.push(t, y.clone(), f_cur.clone());
                }

                t = res.t;
                y = res.y;
                f_cur = res.f_new;
                h = next;
                steps += 1;
                if steps > max_steps {
                    return Err(OdeError::MaxSteps {
                        t_final,
                        max_steps,
                    });
                }
                if dir * (t_final - t) <= 0.0 {
                    record_final(t, &y, t_eval, &mut eval_idx, t_final, dir, &mut outputs);
                    break;
                }
            }
            Err(OdeError::Newton { t: _, residual: _ })
            | Err(OdeError::StepTooSmall { t: _ }) => {
                let contracted = h * 0.5;
                if contracted.abs() < h_min {
                    return Err(OdeError::StepTooSmall { t });
                }
                h = contracted;
            }
            Err(e) => return Err(e),
        }
    }

    if t_eval.is_none() {
        return Ok(vec![y]);
    }
    Ok(outputs)
}

fn record_outputs(
    t_old: f64,
    y_old: &[f64],
    f_old: &[f64],
    res: &StepResult,
    t_eval: Option<&[f64]>,
    eval_idx: &mut usize,
    dir: f64,
    outputs: &mut Vec<Vec<f64>>,
) {
    if let Some(te) = t_eval {
        while *eval_idx < te.len() && dir * (te[*eval_idx] - t_old) <= dir * (res.t - t_old) {
            let tev = te[*eval_idx];
            outputs.push(hermite(t_old, y_old, f_old, res.t, &res.y, &res.f_new, tev));
            *eval_idx += 1;
        }
    }
}

fn record_final(
    t: f64,
    y: &[f64],
    t_eval: Option<&[f64]>,
    eval_idx: &mut usize,
    t_final: f64,
    dir: f64,
    outputs: &mut Vec<Vec<f64>>,
) {
    if let Some(te) = t_eval {
        while *eval_idx < te.len() && dir * (te[*eval_idx] - t) <= 0.0 {
            outputs.push(y.to_vec());
            *eval_idx += 1;
        }
        if *eval_idx < te.len() && (te[*eval_idx] - t_final).abs() < 1e-12 {
            outputs.push(y.to_vec());
            *eval_idx += 1;
        }
    } else {
        let _ = (t, t_final);
    }
}

/// Cubic Hermite interpolation at `tau ∈ [t0, t1]`.
fn hermite(
    t0: f64,
    y0: &[f64],
    f0: &[f64],
    t1: f64,
    y1: &[f64],
    f1: &[f64],
    tau: f64,
) -> Vec<f64> {
    let h = t1 - t0;
    let s = (tau - t0) / h;
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;
    let mut out = vec![0.0; y0.len()];
    for i in 0..y0.len() {
        out[i] = h00 * y0[i] + h10 * h * f0[i] + h01 * y1[i] + h11 * h * f1[i];
    }
    out
}

fn weighted_norm(err: &[f64], y: &[f64], rtol: f64, atol: f64) -> f64 {
    let n = err.len();
    let mut sum = 0.0;
    for i in 0..n {
        let w = atol + rtol * y[i].abs();
        sum += (err[i] / w).powi(2);
    }
    (sum / n as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Tsit45 — explicit Runge–Kutta (Tsitouras 4(5)), non-stiff.
// ---------------------------------------------------------------------------

const TSIT_C: [f64; 6] = [0.0, 0.161, 0.327, 0.9, 0.9800251915382129, 1.0];
const TSIT_A: [[f64; 5]; 5] = [
    [0.161, 0.0, 0.0, 0.0, 0.0],
    [-0.0084806554918121, 0.3354806554918121, 0.0, 0.0, 0.0],
    [2.897153057105493, -6.359448489975075, 4.362295432869582, 0.0, 0.0],
    [5.145377361938561, -11.21379794921784, 10.46437040342962, -3.938106759693844, 0.0],
    [-8.897713934953242, 17.94341276291233, -15.11813299877088, 6.328291979769248, -0.2508974955034598],
];
const TSIT_B: [f64; 6] = [0.1570248978603244, 0.0, 0.3275391890126924, 0.2600303549540808, -0.1427533716485565, 0.04872719044648867];
const TSIT_BHAT: [f64; 6] = [0.09493171702, 0.0, 0.24128936911902733, 0.25030664815569845, -0.06280919694580709, 0.06199905792219885];

fn step_tsit45(
    f: &RhsFn,
    t: f64,
    y: &[f64],
    h: f64,
    _rtol: f64,
    _atol: f64,
) -> Result<StepResult, OdeError> {
    let n = y.len();
    let mut k = vec![vec![0.0; n]; 6];
    k[0] = eval(f, t, y);
    let mut stage = vec![y.to_vec(); 6];
    for s in 1..6 {
        for i in 0..n {
            let mut acc = y[i];
            for col in 0..s {
                let a = TSIT_A[s - 1][col];
                if a != 0.0 {
                    acc += h * a * k[col][i];
                }
            }
            stage[s][i] = acc;
        }
        k[s] = eval(f, t + h * TSIT_C[s], &stage[s]);
    }
    let mut y_new = vec![0.0; n];
    let mut y_hat = vec![0.0; n];
    for i in 0..n {
        let mut acc5 = y[i];
        let mut acc4 = y[i];
        for s in 0..6 {
            acc5 += h * TSIT_B[s] * k[s][i];
            acc4 += h * TSIT_BHAT[s] * k[s][i];
        }
        y_new[i] = acc5;
        y_hat[i] = acc4;
    }
    let err = y_new.iter().zip(&y_hat).map(|(a, b)| a - b).collect::<Vec<_>>();
    let f_new = eval(f, t + h, &y_new);
    Ok(StepResult {
        t: t + h,
        y: y_new,
        err,
        f_new,
    })
}

// ---------------------------------------------------------------------------
// TR-BDF2 — 2-stage SDIRK, A-stable, stiff.
// ---------------------------------------------------------------------------

const TRBDF2_GAMMA: f64 = 0.2928932188134524; // 2 - sqrt(2)

fn step_trbdf2(
    f: &RhsFn,
    t: f64,
    y: &[f64],
    h: f64,
    _rtol: f64,
    _atol: f64,
) -> Result<StepResult, OdeError> {
    let n = y.len();
    let g = TRBDF2_GAMMA;
    let f_start = eval(f, t, y);

    // Stage 1: k1 = f(t + g·h, y + h·g·k1)
    let (k1, _y1) = implicit_stage(f, t + g * h, y, y, g * h, &f_start, &f_start)?;
    // Stage 2: k2 = f(t + h, y + h·g·k1 + h·(1-2g)·k2)
    let base2: Vec<f64> = y
        .iter()
        .zip(&k1)
        .map(|(yi, k1i)| yi + h * g * k1i)
        .collect();
    let (k2, y2) = implicit_stage(f, t + h, &base2, y, (1.0 - 2.0 * g) * h, &f_start, &k1)?;

    let y_new = y2.clone();
    let mut err = vec![0.0; n];
    for i in 0..n {
        let y_trap = y[i] + 0.5 * h * (k1[i] + k2[i]);
        err[i] = y_new[i] - y_trap;
    }
    let f_new = k2.clone();
    Ok(StepResult {
        t: t + h,
        y: y_new,
        err,
        f_new,
    })
}

/// Solve one SDIRK stage `k = f(t_stage, y_base + h·diag·k)` via Newton. `f_start`
/// seeds the predictor (explicit Euler from `seed_base`), and `seed_k` is the
/// previous stage's derivative. Returns the stage derivative `k` and the stage
/// state `y_stage`.
fn implicit_stage(
    f: &RhsFn,
    t_stage: f64,
    y_base: &[f64],
    seed_base: &[f64],
    diag: f64,
    f_start: &[f64],
    seed_k: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), OdeError> {
    let n = y_base.len();
    let mut k = seed_k.to_vec();
    let mut y_stage: Vec<f64> = seed_base
        .iter()
        .zip(seed_k)
        .map(|(b, kk)| b + diag * kk)
        .collect();
    let _ = f_start;
    let fk0 = eval(f, t_stage, &y_stage);
    let jac = jacobian(f, t_stage, &y_stage, &fk0);
    for _iter in 0..20 {
        let fk = eval(f, t_stage, &y_stage);
        let r: Vec<f64> = k.iter().zip(&fk).map(|(kk, fv)| kk - fv).collect();
        let res_norm = norm2(&r);
        if res_norm < 1e-12 {
            break;
        }
        let delta =
            solve_newton_system(&jac, diag, &r).ok_or(OdeError::Newton { t: t_stage, residual: res_norm })?;
        for i in 0..n {
            k[i] -= delta[i];
            y_stage[i] = y_base[i] + diag * k[i];
        }
    }
    Ok((k, y_stage))
}

// ---------------------------------------------------------------------------
// ESDIRK34 — 4-stage ESDIRK order 3(4), Jørgensen, Kristensen & Thomsen (2018),
// arXiv:1803.01613, Table 3.1.
// ---------------------------------------------------------------------------

const ESDIRK34_GAMMA: f64 = 0.43586652150845899942;
const ESDIRK34_C2: f64 = 0.87173304301691799883;
const ESDIRK34_C3: f64 = 0.46823874485184439565;
const ESDIRK34_A31: f64 = 0.14073777472470619619;
const ESDIRK34_A32: f64 = -0.1083655513813208000;
const ESDIRK34_B: [f64; 4] = [
    0.10239940061991099768,
    -0.368784522555561061,
    0.83861253012718610911,
    0.43586652150845899942,
];
const ESDIRK34_BHAT: [f64; 4] = [
    0.15702489786032493710,
    0.11733044137043884870,
    0.61667803039212146434,
    0.10896663037711474985,
];

fn step_esdirk34(
    f: &RhsFn,
    t: f64,
    y: &[f64],
    h: f64,
    _rtol: f64,
    _atol: f64,
) -> Result<StepResult, OdeError> {
    let n = y.len();
    let g = ESDIRK34_GAMMA;
    let mut k = vec![vec![0.0; n]; 4];
    k[0] = eval(f, t, y); // explicit first stage

    let (k1, _y1) = implicit_stage(f, t + ESDIRK34_C2 * h, y, y, ESDIRK34_A21_H * h, &k[0], &k[0])?;
    k[1] = k1;
    let base2 = y
        .iter()
        .zip(&k[0])
        .zip(&k[1])
        .map(|((yi, k0i), k1i)| yi + h * (ESDIRK34_A31 * k0i + ESDIRK34_A32 * k1i))
        .collect::<Vec<_>>();
    let (k2, _y2) = implicit_stage(f, t + ESDIRK34_C3 * h, &base2, y, g * h, &k[0], &k[1])?;
    k[2] = k2;
    let base3 = y
        .iter()
        .zip(&k[0])
        .zip(&k[1])
        .zip(&k[2])
        .map(|(((yi, k0i), k1i), k2i)| {
            yi + h * (ESDIRK34_B[0] * k0i + ESDIRK34_B[1] * k1i + ESDIRK34_B[2] * k2i)
        })
        .collect::<Vec<_>>();
    let (k3, _y3) = implicit_stage(f, t + h, &base3, y, g * h, &k[0], &k[2])?;
    k[3] = k3;

    let mut y_new = vec![0.0; n];
    let mut y_hat = vec![0.0; n];
    for i in 0..n {
        let mut acc3 = y[i];
        let mut acc4 = y[i];
        for s in 0..4 {
            acc3 += h * ESDIRK34_B[s] * k[s][i];
            acc4 += h * ESDIRK34_BHAT[s] * k[s][i];
        }
        y_new[i] = acc3;
        y_hat[i] = acc4;
    }
    let err = y_new.iter().zip(&y_hat).map(|(a, b)| a - b).collect::<Vec<_>>();
    let f_new = k[3].clone();
    Ok(StepResult {
        t: t + h,
        y: y_new,
        err,
        f_new,
    })
}

const ESDIRK34_A21_H: f64 = 0.43586652150845899942; // a21 == gamma

// ---------------------------------------------------------------------------
// BDF — variable-order (1–5) backward differentiation, stiff.
// ---------------------------------------------------------------------------

const BDF_ALPHA: [[f64; 6]; 5] = [
    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, -4.0 / 3.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
    [1.0, -18.0 / 11.0, 9.0 / 11.0, -2.0 / 11.0, 0.0, 0.0],
    [1.0, -48.0 / 25.0, 36.0 / 25.0, -16.0 / 25.0, 3.0 / 25.0, 0.0],
    [1.0, -300.0 / 137.0, 300.0 / 137.0, -200.0 / 137.0, 75.0 / 137.0, -12.0 / 137.0],
];
const BDF_BETA0: [f64; 5] = [1.0, 2.0 / 3.0, 6.0 / 11.0, 12.0 / 25.0, 60.0 / 137.0];

/// BDF integrator state: recent (t, y, f) triples, most-recent first, plus the
/// current order.
struct BdfState {
    hist: Vec<(f64, Vec<f64>, Vec<f64>)>,
    order: usize,
}

impl BdfState {
    fn new() -> Self {
        BdfState {
            hist: Vec::new(),
            order: 1,
        }
    }
    fn push(&mut self, t: f64, y: Vec<f64>, fy: Vec<f64>) {
        self.hist.insert(0, (t, y, fy));
        if self.hist.len() > 6 {
            self.hist.truncate(6); // enough for order 5 (needs 5 previous)
        }
    }
}

fn step_bdf(
    f: &RhsFn,
    t: f64,
    y: &[f64],
    h: f64,
    _rtol: f64,
    _atol: f64,
    mut state: Option<&mut BdfState>,
) -> Result<StepResult, OdeError> {
    let n = y.len();
    let mut st = state.as_deref_mut();
    // Choose order from available history (need `order` previous points).
    let order = match st.as_deref() {
        Some(s) => {
            let max_k = s.hist.len().min(5);
            s.order.min(max_k).max(1)
        }
        None => 1,
    };

    let alpha = &BDF_ALPHA[order - 1];
    let beta0 = BDF_BETA0[order - 1];

    // Predictor: extrapolate recent polynomial to t+h.
    let y_pred = bdf_predict(y, h, order, st.as_deref());

    let f_new = eval(f, t + h, &y_pred);
    let jac = jacobian(f, t + h, &y_pred, &f_new);
    let gamma = h * beta0;

    let mut y_cur = y_pred.clone();
    for _iter in 0..20 {
        let fk = eval(f, t + h, &y_cur);
        let mut residual = vec![0.0; n];
        for i in 0..n {
            let mut acc = y_cur[i];
            for j in 1..=order {
                // history is most-recent-first; index j-1 is the j-th previous.
                if let Some(prev) = st.as_ref().and_then(|s| s.hist.get(j - 1)) {
                    acc += alpha[j] * prev.1[i];
                }
            }
            residual[i] = acc - gamma * fk[i];
        }
        let res_norm = norm2(&residual);
        if res_norm < 1e-12 {
            break;
        }
        let delta = solve_newton_system(&jac, gamma, &residual)
            .ok_or(OdeError::Newton { t: t + h, residual: res_norm })?;
        for i in 0..n {
            y_cur[i] -= delta[i];
        }
    }

    // Error estimate: difference between order-k and order-(k-1) correctors.
    let err = if let Some(s) = st.as_deref() {
        if order > 1 && s.hist.len() >= order {
            let prev = bdf_predict(y, h, order - 1, Some(s));
            y_cur
                .iter()
                .zip(&prev)
                .map(|(a, b)| (a - b) * (order as f64 / (order as f64 + 1.0)))
                .collect::<Vec<_>>()
        } else {
            vec![0.0; n]
        }
    } else {
        vec![0.0; n]
    };

    // Order control: prefer to raise order when history permits.
    if let Some(s) = st.as_deref_mut() {
        if order < 5 && s.hist.len() >= order + 1 {
            s.order = order + 1;
        } else if order > 1 {
            s.order = order.min(s.hist.len());
        }
    }

    let f_new = eval(f, t + h, &y_cur);
    Ok(StepResult {
        t: t + h,
        y: y_cur,
        err,
        f_new,
    })
}

/// Extrapolate the recent-state polynomial (stored most-recent-first in
/// `state.hist`) to `t + h` using backward differences. For the first step
/// (no history) it falls back to the current `y`.
fn bdf_predict(y: &[f64], h: f64, order: usize, state: Option<&BdfState>) -> Vec<f64> {
    let st = match state {
        Some(s) if !s.hist.is_empty() => s,
        _ => return y.to_vec(),
    };
    let dt = (st.hist[0].0 - st.hist[1].0).abs();
    if dt <= 0.0 || order <= 1 || st.hist.len() < order {
        return y.to_vec();
    }
    let mut out = st.hist[0].1.clone();
    let mut diffs = st.hist[0].1.clone();
    let mut scale = 1.0;
    for p in 1..order {
        if st.hist.len() <= p {
            break;
        }
        let mut next = vec![0.0; diffs.len()];
        for i in 0..diffs.len() {
            next[i] = st.hist[p - 1].1[i] - st.hist[p].1[i];
        }
        scale *= (h / dt) / (p as f64);
        for i in 0..out.len() {
            out[i] += scale * next[i];
        }
        diffs = next;
    }
    out
}

impl OdeProblem {
    /// Integrate the problem from its initial time to `t_final` and return the
    /// state vector at `t_final`, using the requested `method`.
    ///
    /// # Errors
    ///
    /// Returns [`OdeError`] if the integration fails (non-convergent Newton
    /// step, collapsed step size, or step budget exceeded).
    pub fn solve(&self, method: Method, t_final: f64) -> Result<Vec<f64>, OdeError> {
        let out = integrate(self, method, t_final, None)?;
        Ok(out.into_iter().next().unwrap_or_else(|| self.y0.clone()))
    }

    /// Integrate at each time in `t_eval` (each strictly increasing and beyond
    /// `t0`) and return one state vector per evaluation time.
    ///
    /// The integrator lands exactly on each `t_eval` point and interpolates with
    /// a cubic Hermite polynomial between accepted steps, so output is exact at
    /// the requested times.
    ///
    /// # Errors
    ///
    /// Returns [`OdeError`] if the integration fails (see [`OdeProblem::solve`]).
    pub fn solve_dense(&self, method: Method, t_eval: &[f64]) -> Result<Vec<Vec<f64>>, OdeError> {
        if t_eval.is_empty() {
            return Ok(Vec::new());
        }
        integrate(self, method, *t_eval.last().unwrap(), Some(t_eval))
    }
}
