use diffsol::{
    DenseMatrix, MatrixCommon, NalgebraLU, NalgebraMat, NalgebraVec, OdeBuilder, OdeSolverMethod,
    VectorView,
};

use crate::error::OdeError;
use crate::problem::{OdeProblem, nalgebra_vec_to_vec};

/// Integration method selection for [`OdeProblem::solve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Backward Differentiation Formulae — variable-order, stiff-capable,
    /// handles singular mass matrices. The default for general problems.
    Bdf,
    /// Explicit (non-stiff) Runge–Kutta (Tsitouras 4(5)). Cheap when the system
    /// is non-stiff.
    Tsit45,
    /// Trapezoidal-rule BDF2 (SDIRK/ESDIRK family) — stiff-capable.
    TrBdf2,
    /// Explicit singly-diagonally-implicit RK of order 3(4) — stiff-capable.
    Esdirk34,
}

/// Build a diffsol `OdeSolverProblem` from an [`OdeProblem`], moving the RHS and
/// initial-condition closures into diffsol's ownership. The exact `Eqn` type is
/// inferred locally (it is an internal diffsol type and is not nameable), which
/// is why this is a macro rather than a function with an explicit return type.
///
/// The RHS is supplied via `rhs_implicit` together with a matrix-free Jacobian
/// vector product `J·v` computed by forward finite differences of the user's
/// RHS, so callers never have to provide an analytic Jacobian.
macro_rules! build_problem {
    ($self:expr) => {{
        let n = $self.nstates;
        let f = $self.rhs.clone();
        let f_jac = $self.rhs.clone();
        let y0 = $self.y0.clone();
        let rhs_closure =
            move |x: &NalgebraVec<f64>, _p: &NalgebraVec<f64>, t: f64, y: &mut NalgebraVec<f64>| {
                let mut xb = vec![0.0; n];
                let mut yb = vec![0.0; n];
                for i in 0..n {
                    xb[i] = x[i];
                }
                f(t, &xb, &mut yb);
                for i in 0..n {
                    y[i] = yb[i];
                }
            };
        let jac_closure = move |x: &NalgebraVec<f64>,
                                _p: &NalgebraVec<f64>,
                                t: f64,
                                v: &NalgebraVec<f64>,
                                y: &mut NalgebraVec<f64>| {
            let mut xb = vec![0.0; n];
            let mut vb = vec![0.0; n];
            for i in 0..n {
                xb[i] = x[i];
                vb[i] = v[i];
            }
            let mut fx = vec![0.0; n];
            f_jac(t, &xb, &mut fx);
            let eps = 1e-8;
            let mut xpv = xb.clone();
            for i in 0..n {
                xpv[i] += eps * vb[i];
            }
            let mut fxpv = vec![0.0; n];
            f_jac(t, &xpv, &mut fxpv);
            for i in 0..n {
                y[i] = (fxpv[i] - fx[i]) / eps;
            }
        };
        let init_closure = move |_p: &NalgebraVec<f64>, _t: f64, y: &mut NalgebraVec<f64>| {
            for i in 0..n {
                y[i] = y0[i];
            }
        };
        OdeBuilder::<NalgebraMat<f64>>::new()
            .t0($self.t0)
            .rtol($self.rtol)
            .atol(vec![$self.atol; n])
            .h0($self.h0)
            .rhs_implicit(rhs_closure, jac_closure)
            .init(init_closure, n)
            .build()
            .map_err(OdeError::from)
    }};
}

impl OdeProblem {
    /// Integrate the problem from its initial time to `t_final` and return the
    /// state vector at `t_final`, using the requested `method`.
    ///
    /// # Errors
    ///
    /// Returns [`OdeError`] if the underlying diffsol problem cannot be built or
    /// the integration fails (e.g. the step does not converge or `t_final` is
    /// invalid).
    pub fn solve(&self, method: Method, t_final: f64) -> Result<Vec<f64>, OdeError> {
        match method {
            Method::Bdf => self.solve_bdf(t_final),
            Method::Tsit45 => self.solve_tsit45(t_final),
            Method::TrBdf2 => self.solve_tr_bdf2(t_final),
            Method::Esdirk34 => self.solve_esdirk34(t_final),
        }
    }

    /// Integrate at each time in `t_eval` (each must be greater than `t0`) and
    /// return one state vector per evaluation time.
    ///
    /// Unlike a naive loop over [`OdeProblem::solve`], this builds the solver
    /// once and walks a *single* trajectory, interpolating the requested times
    /// off that one integration. This turns the O(n) redundant re-integration
    /// from `t0` (one full solve per evaluation point) into a single solve plus
    /// cheap interpolations, which matters for trajectory plotting with many
    /// `t_eval` points.
    ///
    /// The entries of `t_eval` must be strictly increasing and greater than or
    /// equal to `t0` (this mirrors the underlying diffsol requirement); a
    /// violating `t_eval` is rejected rather than silently mis-integrated.
    ///
    /// # Errors
    ///
    /// Returns [`OdeError`] if the integration fails or `t_eval` is not a valid
    /// strictly-increasing sequence starting at/after `t0` (see
    /// [`OdeProblem::solve`]).
    pub fn solve_dense(&self, method: Method, t_eval: &[f64]) -> Result<Vec<Vec<f64>>, OdeError> {
        if t_eval.is_empty() {
            return Ok(Vec::new());
        }
        let n = self.nstates;
        let mut out = Vec::with_capacity(t_eval.len());
        match method {
            Method::Bdf => {
                let problem = build_problem!(self)?;
                let mut solver = problem.bdf::<NalgebraLU<f64>>().map_err(OdeError::from)?;
                let (matrix, _stop) = solver.solve_dense(t_eval).map_err(OdeError::from)?;
                for j in 0..matrix.ncols() {
                    let col = matrix.column(j).into_owned();
                    out.push(nalgebra_vec_to_vec(&col, n));
                }
            }
            Method::Tsit45 => {
                let problem = build_problem!(self)?;
                let mut solver = problem.tsit45().map_err(OdeError::from)?;
                let (matrix, _stop) = solver.solve_dense(t_eval).map_err(OdeError::from)?;
                for j in 0..matrix.ncols() {
                    let col = matrix.column(j).into_owned();
                    out.push(nalgebra_vec_to_vec(&col, n));
                }
            }
            Method::TrBdf2 => {
                let problem = build_problem!(self)?;
                let mut solver = problem
                    .tr_bdf2::<NalgebraLU<f64>>()
                    .map_err(OdeError::from)?;
                let (matrix, _stop) = solver.solve_dense(t_eval).map_err(OdeError::from)?;
                for j in 0..matrix.ncols() {
                    let col = matrix.column(j).into_owned();
                    out.push(nalgebra_vec_to_vec(&col, n));
                }
            }
            Method::Esdirk34 => {
                let problem = build_problem!(self)?;
                let mut solver = problem
                    .esdirk34::<NalgebraLU<f64>>()
                    .map_err(OdeError::from)?;
                let (matrix, _stop) = solver.solve_dense(t_eval).map_err(OdeError::from)?;
                for j in 0..matrix.ncols() {
                    let col = matrix.column(j).into_owned();
                    out.push(nalgebra_vec_to_vec(&col, n));
                }
            }
        }
        Ok(out)
    }

    /// BDF solve to a single final time. See [`Method::Bdf`].
    ///
    /// # Errors
    ///
    /// Returns [`OdeError`] if the diffsol problem cannot be built or the
    /// integration fails.
    pub fn solve_bdf(&self, t_final: f64) -> Result<Vec<f64>, OdeError> {
        let n = self.nstates;
        let problem = build_problem!(self)?;
        let mut solver = problem.bdf::<NalgebraLU<f64>>().map_err(OdeError::from)?;
        solver.solve(t_final).map_err(OdeError::from)?;
        let v = solver.interpolate(t_final).map_err(OdeError::from)?;
        Ok(nalgebra_vec_to_vec(&v, n))
    }

    /// Explicit RK (Tsitouras 4(5)) solve to a single final time.
    ///
    /// # Errors
    ///
    /// Returns [`OdeError`] if the diffsol problem cannot be built or the
    /// integration fails.
    pub fn solve_tsit45(&self, t_final: f64) -> Result<Vec<f64>, OdeError> {
        let n = self.nstates;
        let problem = build_problem!(self)?;
        let mut solver = problem.tsit45().map_err(OdeError::from)?;
        solver.solve(t_final).map_err(OdeError::from)?;
        let v = solver.interpolate(t_final).map_err(OdeError::from)?;
        Ok(nalgebra_vec_to_vec(&v, n))
    }

    /// TR-BDF2 solve to a single final time.
    ///
    /// # Errors
    ///
    /// Returns [`OdeError`] if the diffsol problem cannot be built or the
    /// integration fails.
    pub fn solve_tr_bdf2(&self, t_final: f64) -> Result<Vec<f64>, OdeError> {
        let n = self.nstates;
        let problem = build_problem!(self)?;
        let mut solver = problem
            .tr_bdf2::<NalgebraLU<f64>>()
            .map_err(OdeError::from)?;
        solver.solve(t_final).map_err(OdeError::from)?;
        let v = solver.interpolate(t_final).map_err(OdeError::from)?;
        Ok(nalgebra_vec_to_vec(&v, n))
    }

    /// ESDIRK34 solve to a single final time.
    ///
    /// # Errors
    ///
    /// Returns [`OdeError`] if the diffsol problem cannot be built or the
    /// integration fails.
    pub fn solve_esdirk34(&self, t_final: f64) -> Result<Vec<f64>, OdeError> {
        let n = self.nstates;
        let problem = build_problem!(self)?;
        let mut solver = problem
            .esdirk34::<NalgebraLU<f64>>()
            .map_err(OdeError::from)?;
        solver.solve(t_final).map_err(OdeError::from)?;
        let v = solver.interpolate(t_final).map_err(OdeError::from)?;
        Ok(nalgebra_vec_to_vec(&v, n))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn exp_decay_bdf() {
        // dy/dt = -y, y(0) = 1  ->  y(t) = e^-t
        let prob = OdeProblem::new(|_t, y, dydt| dydt[0] = -y[0], vec![1.0], 0.0).unwrap();
        let y = prob.solve(Method::Bdf, 1.0).unwrap();
        assert_abs_diff_eq!(y[0], std::f64::consts::E.recip(), epsilon = 1e-4);
    }

    #[test]
    fn exp_decay_all_methods() {
        // dy/dt = 2y, y(0) = 1  ->  y(t) = e^(2t)
        let prob = OdeProblem::new(|_t, y, dydt| dydt[0] = 2.0 * y[0], vec![1.0], 0.0).unwrap();
        for m in [
            Method::Bdf,
            Method::Tsit45,
            Method::TrBdf2,
            Method::Esdirk34,
        ] {
            let y = prob.solve(m, 0.5).unwrap();
            assert_abs_diff_eq!(y[0], (2.0 * 0.5_f64).exp(), epsilon = 1e-3);
        }
    }

    #[test]
    fn harmonic_oscillator() {
        // y' = z, z' = -y,  (y,z)(0) = (1,0)  ->  (cos t, -sin t)
        let prob = OdeProblem::new(
            |_t, s, ds| {
                ds[0] = s[1];
                ds[1] = -s[0];
            },
            vec![1.0, 0.0],
            0.0,
        )
        .unwrap();
        let y = prob
            .solve(Method::Bdf, std::f64::consts::FRAC_PI_2)
            .unwrap();
        assert_abs_diff_eq!(y[0], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(y[1], -1.0, epsilon = 1e-3);
    }

    #[test]
    fn dense_eval_matches_point_eval() {
        let prob = OdeProblem::new(|_t, y, dydt| dydt[0] = -y[0], vec![1.0], 0.0).unwrap();
        let dense = prob.solve_dense(Method::Bdf, &[0.5, 1.0, 1.5]).unwrap();
        assert_eq!(dense.len(), 3);
        assert_abs_diff_eq!(dense[0][0], (-0.5_f64).exp(), epsilon = 1e-4);
        assert_abs_diff_eq!(dense[1][0], (-1.0_f64).exp(), epsilon = 1e-4);
        assert_abs_diff_eq!(dense[2][0], (-1.5_f64).exp(), epsilon = 1e-4);
    }

    #[test]
    fn negative_initial_time() {
        let prob = OdeProblem::new(|_t, y, dydt| dydt[0] = y[0], vec![1.0], -1.0).unwrap();
        let y = prob.solve(Method::Bdf, 0.0).unwrap();
        // one unit of growth from t=-1 to t=0
        assert_abs_diff_eq!(y[0], std::f64::consts::E, epsilon = 1e-3);
    }

    #[test]
    fn empty_initial_state_is_rejected() {
        let res = OdeProblem::new(|_t, _y, _dydt| {}, vec![], 0.0);
        assert!(matches!(res, Err(OdeError::Invalid(_))));
    }
}
