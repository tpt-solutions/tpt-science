//! # Classical (soft-matter) DFT tour: surface tension vs temperature
//!
//! `tpt-sci-dft-classical` wraps the [`feos`] framework ([`feos_dft`]) and adds a
//! tiny [`ClassicalDft`] handle plus a [`DftError`] type. The companion example
//! (`examples/adsorption.rs`) solves a single planar vapour-liquid interface and
//! then a slit-pore **adsorption isotherm** (a pressure sweep). This example
//! instead exercises a *different* corner of the same public surface: the
//! [`feos_dft::interface::SurfaceTensionDiagram`], which solves a whole stack of
//! planar interfacial density profiles across a **temperature sweep** and returns
//! the surface tension `γ(T)` and the interfacial (90/10) thickness in one go.
//!
//! Classical DFT minimises the grand potential
//!
//! ```text
//! Omega[rho] = F[rho] + integral rho(r) * (V_ext(r) - mu) dr
//! ```
//!
//! over an inhomogeneous density profile; the excess grand potential of a planar
//! interface *is* its surface tension. Sweeping `T` lets us watch `γ` fall
//! towards zero as the critical temperature `T_c` is approached (the interface
//! thickens and finally disappears) — the hallmark of critical-point scaling.
//!
//! ## What to observe when running
//!
//! 1. Bulk reference states: the PC-SAFT critical point and the saturation
//!    (vapour-liquid equilibrium) at a sequence of sub-critical temperatures.
//! 2. For each `T`: the saturation pressure, the coexisting vapour/liquid bulk
//!    densities, the surface tension `γ(T)` and the 90/10 interfacial thickness.
//! 3. Critical-point physics: `γ` is largest in the cold, dense limit and
//!    decreases monotonically toward `T_c`, while the interface thickens — the
//!    two move in opposite directions as criticality is approached.
//!
//! Every fallible `feos` call is funnelled through [`DftError`], so `main`
//! returns `Result<(), DftError>`. As in the adsorption example the PC-SAFT
//! record for propane is built in code (no external data files), and the grid is
//! deliberately modest (512 interface points) so the tour runs in seconds.
//!
//! Run with: `cargo run --release --example isotherm -p tpt-sci-dft-classical`

use std::fmt::Display;

use feos::pcsaft::{PcSaftFunctional, PcSaftParameters, PcSaftRecord};
use feos_core::parameter::{Identifier, PureRecord};
use feos_core::{Contributions, PhaseEquilibrium, ReferenceSystem, State};
use feos_dft::interface::SurfaceTensionDiagram;
use tpt_sci_dft_classical::{ClassicalDft, DftError};

/// Grid points across each planar vapour-liquid interface (keep modest: DFT is
/// FFT-convolution bound and `feos`' own tests use 2048).
const INTERFACE_POINTS: usize = 512;
/// Width of each interface domain in Angstrom (feos' internal reduced length).
const INTERFACE_WIDTH: f64 = 100.0;
/// Sub-critical temperature fractions `T / T_c` sampled by the diagram. They
/// stay safely below the critical point, where the solver is robust; the
/// critical temperature itself is excluded because the interface then vanishes.
const TEMPERATURE_FRACTIONS: [f64; 7] = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90];

/// Interpret `value` in `feos`' internal *reduced* unit system and return the
/// corresponding `quantity` value (concrete unit inferred from the call site).
fn from_reduced<Q: ReferenceSystem<Inner = f64>>(value: f64) -> Q {
    Q::from_reduced(value)
}

/// Wrap a `feos` parameter/functional failure in the crate's error type.
fn functional_error(error: impl Display) -> DftError {
    DftError::Functional(error.to_string())
}

/// PC-SAFT parameters for propane, assembled in code (Gross & Sadowski 2001
/// style record: segment number, segment diameter, dispersion energy).
fn propane_parameters() -> Result<PcSaftParameters, DftError> {
    let identifier = Identifier::new(
        Some("74-98-6"),
        Some("propane"),
        Some("propane"),
        Some("CCC"),
        None,
        Some("C3H8"),
    );
    let record = PcSaftRecord::new(2.001_829, 3.618_353, 208.110_1, 0.0, 0.0, None, None, None);
    let pure = PureRecord::new(identifier, 44.096_2, record);
    PcSaftParameters::new_pure(pure).map_err(functional_error)
}

fn main() -> Result<(), DftError> {
    println!("=== tpt-sci-dft-classical: surface-tension temperature scan (PC-SAFT) ===\n");

    // ---------------------------------------------------------------- [1/3] --
    println!("[1/3] parameters + Helmholtz energy functional");
    let functional = PcSaftFunctional::new(propane_parameters()?);
    // The wrapper owns the functional behind an `Arc`; `functional_ref` hands
    // back the borrow that every `feos`/`feos_dft` entry point expects.
    let dft = ClassicalDft::with_functional(&functional);
    let func = dft.functional_ref();
    let record = &dft.functional.parameters.pure[0].model_record;
    println!(
        "      PC-SAFT propane: components = {}, m = {:.4}, sigma = {:.4} A, eps/k = {:.2} K",
        dft.functional.parameters.pure.len(),
        record.m,
        record.sigma,
        record.epsilon_k
    );

    // ---------------------------------------------------------------- [2/3] --
    println!("\n[2/3] bulk reference states across a temperature sweep");
    let critical: State<_> = State::critical_point(func, (), None, None, Default::default())
        .map_err(functional_error)?;
    let critical_temperature = critical.temperature;
    let t_c_red = critical_temperature.to_reduced();
    println!(
        "      critical point: T_c = {}, p_c = {}, rho_c = {}",
        critical_temperature,
        critical.pressure(Contributions::Total),
        critical.density
    );

    // Build one vapour-liquid equilibrium per requested sub-critical temperature.
    let mut vles = Vec::with_capacity(TEMPERATURE_FRACTIONS.len());
    for &frac in &TEMPERATURE_FRACTIONS {
        // `critical_temperature` is a `Temperature`; `Temperature * f64` yields a
        // `Temperature`, exactly as in the adsorption example's saturation state.
        let temperature = critical_temperature * frac;
        let vle = PhaseEquilibrium::pure(func, temperature, None, Default::default())
            .map_err(functional_error)?;
        println!(
            "      T = {:.3} K (T/Tc = {:.2}): p_sat = {}, rho_vap = {}, rho_liq = {}",
            temperature,
            frac,
            vle.vapor().pressure(Contributions::Total),
            vle.vapor().density,
            vle.liquid().density
        );
        vles.push(vle);
    }
    assert!(
        !vles.is_empty(),
        "at least one coexistence state must be available"
    );

    // ---------------------------------------------------------------- [3/3] --
    println!("\n[3/3] surface-tension diagram γ(T) and interfacial thickness");
    // `SurfaceTensionDiagram::new` solves every interface (initialising the
    // single-component profiles with pDGT, then tanh for mixtures) and drops any
    // that fail to converge — so we assert on the converged subset below.
    let mut diagram = SurfaceTensionDiagram::new(
        &vles,
        None,
        Some(INTERFACE_POINTS),
        Some(from_reduced(INTERFACE_WIDTH)),
        Some(critical_temperature),
        None,
        None,
    );

    let gamma = diagram.surface_tension();
    let thickness = diagram.interfacial_thickness();
    let n = diagram.profiles.len();
    println!(
        "  {:>8} {:>10} {:>14} {:>14} {:>12} {:>12}",
        "T/Tc", "T[K]", "p_sat", "gamma", "thickness", "rho_liq/rho_vap"
    );

    let mut g_vals: Vec<f64> = Vec::with_capacity(n);
    let mut t_vals: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let p = &diagram.profiles[i];
        let tc = p.vle.vapor().temperature.to_reduced() / t_c_red;
        let t = p.vle.vapor().temperature;
        let p_sat = p.vle.vapor().pressure(Contributions::Total);
        let g = gamma.get(i).to_reduced();
        let thick = thickness.get(i).to_reduced();
        let ratio = p.vle.liquid().density.to_reduced() / p.vle.vapor().density.to_reduced();

        println!(
            "  {:>8.2} {:>10.3} {:>14.6} {:>14.6} {:>12.4} {:>12.2}",
            tc, t, p_sat, g, thick, ratio
        );

        // Every density profile must be a finite, non-negative number.
        let density = p.profile.density.to_reduced();
        assert!(
            density.iter().all(|rho| rho.is_finite() && *rho >= 0.0),
            "density profile must be finite and non-negative"
        );
        assert!(g.is_finite() && g > 0.0, "surface tension must be positive");
        assert!(thick.is_finite() && thick > 0.0, "thickness must be positive");
        g_vals.push(g);
        t_vals.push(thick);
    }

    assert!(
        g_vals.len() >= 2,
        "the diagram needs at least two converged profiles to be meaningful"
    );
    // γ falls toward Tc; the interface thickens toward Tc. They move in opposite
    // directions, the signature of critical-point scaling.
    assert!(
        g_vals[0] > g_vals[g_vals.len() - 1],
        "surface tension must decrease as T -> Tc"
    );
    assert!(
        t_vals[0] < t_vals[t_vals.len() - 1],
        "interfacial thickness must increase as T -> Tc"
    );
    println!(
        "\nAll classical-DFT checks passed: γ decreases ({:.4} -> {:.4}) while \
         thickness grows ({:.3} -> {:.3}) toward Tc.",
        g_vals[0],
        g_vals[g_vals.len() - 1],
        t_vals[0],
        t_vals[t_vals.len() - 1]
    );
    Ok(())
}
