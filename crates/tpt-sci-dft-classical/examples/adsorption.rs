//! # Classical (soft-matter) DFT tour: PC-SAFT profiles, surface tension, adsorption
//!
//! `tpt-sci-dft-classical` wraps the [`feos`] framework ([`feos_dft`]) and adds a
//! tiny [`ClassicalDft`] handle plus a [`DftError`] type. This example walks the
//! public surface that a soft-matter DFT user actually touches.
//!
//! ## What classical DFT does here
//!
//! Classical density functional theory minimises the grand potential
//!
//! ```text
//! Omega[rho] = F[rho] + integral rho(r) * (V_ext(r) - mu) dr
//! ```
//!
//! over an inhomogeneous one-body density profile `rho(r)`. The intrinsic
//! Helmholtz energy functional `F[rho]` is built from **PC-SAFT**: hard-sphere
//! FMT, chain and dispersion contributions with their weighted densities linked
//! by FFT convolutions. Solving the Euler-Lagrange equation with Picard and
//! Anderson mixing then yields:
//!
//! * a **vapour-liquid interface** profile — its excess grand potential is the
//!   **surface tension**, and the 90/10 width is the interface thickness;
//! * a **confined (slit pore)** profile under an external wall potential — its
//!   integral is the **adsorbed amount**, so sweeping the bulk pressure traces
//!   an **adsorption isotherm**.
//!
//! ## What to observe when running
//!
//! 1. Bulk reference states: PC-SAFT critical point and the saturation state at
//!    `T = 0.6 * T_c` (propane).
//! 2. The planar profile interpolates monotonically between the liquid and the
//!    vapour bulk density; surface tension is a few tens of mN/m.
//! 3. Pore diagnostics: the adsorbed amount, the grand potential, the wall
//!    (interfacial) tension and the isosteric enthalpy of adsorption of a fluid
//!    confined between two attractive 9-3 Lennard-Jones walls. The contact layer
//!    is enriched by orders of magnitude over the bulk vapour.
//! 4. The isotherm loading grows monotonically with `p/p_sat` (film growth below
//!    capillary condensation).
//!
//! Every fallible `feos` call is funnelled through [`DftError`], so `main`
//! returns `Result<(), DftError>` and both error variants are exercised.
//!
//! No data files are needed: the PC-SAFT record for propane is built in code
//! (the optional `feos` parameter-JSON path is only used to *demonstrate* the
//! `Result` returned by `PcSaftParameters::from_json`). Grids are deliberately
//! modest (512 interface points, 256 pore points, 4 isotherm pressures) so the
//! whole tour runs in seconds instead of minutes; refine them for production
//! numbers.
//!
//! Run with: `cargo run --release --example adsorption -p tpt-sci-dft-classical`

use std::fmt::Display;

use feos::pcsaft::{PcSaftFunctional, PcSaftParameters, PcSaftRecord};
use feos_core::parameter::{Identifier, IdentifierOption, PureRecord};
use feos_core::{
    Contributions, DensityInitialization, PhaseEquilibrium, ReferenceSystem, State, Verbosity,
};
use feos_dft::adsorption::{Adsorption1D, ExternalPotential, Pore1D, PoreSpecification};
use feos_dft::interface::PlanarInterface;
use feos_dft::{DFTSolver, Geometry};
use tpt_sci_dft_classical::{ClassicalDft, DftError};

/// Grid points across the planar vapour-liquid interface (keep modest: DFT is
/// FFT-convolution bound and `feos`' own tests use 2048).
const INTERFACE_POINTS: usize = 512;
/// Width of the interface domain in Angstrom (feos' internal reduced length).
const INTERFACE_WIDTH: f64 = 100.0;
/// Grid points across the slit pore.
const PORE_POINTS: usize = 256;
/// Slit width in Angstrom.
const PORE_WIDTH: f64 = 20.0;
/// Reduced pressures `p / p_sat` sampled by the adsorption isotherm. They stay
/// below capillary condensation of this slit, where the solver is robust and
/// the loading grows monotonically.
const RELATIVE_PRESSURES: [f64; 4] = [0.10, 0.15, 0.22, 0.30];

/// Interpret `value` in `feos`' internal *reduced* unit system (Angstrom for
/// lengths, Kelvin for temperatures, ...) and return the corresponding
/// `quantity` value.
///
/// The `quantity` crate is not a direct dependency of this crate, so the
/// conversion goes through the publicly exposed [`ReferenceSystem`] trait; the
/// concrete unit is inferred from the call site.
fn from_reduced<Q: ReferenceSystem<Inner = f64>>(value: f64) -> Q {
    Q::from_reduced(value)
}

/// Wrap a `feos` parameter/functional failure in the crate's error type.
fn functional_error(error: impl Display) -> DftError {
    DftError::Functional(error.to_string())
}

/// Wrap a `feos_dft` profile/solver failure in the crate's error type.
fn profile_error(error: impl Display) -> DftError {
    DftError::Profile(error.to_string())
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
    // m, sigma / A, epsilon_k / K, dipole moment, quadrupole moment, then the
    // optional entropy-scaling coefficients (viscosity / diffusion / lambda).
    let record = PcSaftRecord::new(2.001_829, 3.618_353, 208.110_1, 0.0, 0.0, None, None, None);
    let pure = PureRecord::new(identifier, 44.096_2, record);
    PcSaftParameters::new_pure(pure).map_err(functional_error)
}

fn main() -> Result<(), DftError> {
    println!("=== tpt-sci-dft-classical: classical DFT tour (PC-SAFT via feos) ===\n");

    // ---------------------------------------------------------------- [1/6] --
    println!("[1/6] parameters + Helmholtz energy functional");

    // `PcSaftParameters::from_json` is the usual entry point but needs feos'
    // vendored parameter files; show the `Result` path either way.
    let json_path = "../../parameters/pcsaft/esper2023.json";
    match PcSaftParameters::from_json(vec!["propane"], json_path, None, IdentifierOption::Name) {
        Ok(_) => println!("      parameters read from {json_path}"),
        Err(e) => println!(
            "      {json_path} unavailable ({}) -> using in-code record",
            functional_error(e)
        ),
    }

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

    // ---------------------------------------------------------------- [2/6] --
    println!("\n[2/6] bulk reference states from the same functional");
    // The annotation pins `State`'s defaults (`Dyn` components, `f64` scalars);
    // `feos` states are generic over dual numbers for automatic differentiation.
    let critical: State<_> = State::critical_point(func, (), None, None, Default::default())
        .map_err(functional_error)?;
    let critical_temperature = critical.temperature;
    println!(
        "      critical point: T_c = {}, p_c = {}, rho_c = {}",
        critical_temperature,
        critical.pressure(Contributions::Total),
        critical.density
    );

    let temperature = critical_temperature * 0.6;
    let vle = PhaseEquilibrium::pure(func, temperature, None, Default::default())
        .map_err(functional_error)?;
    let p_sat = vle.vapor().pressure(Contributions::Total);
    let rho_vapor = vle.vapor().density;
    let rho_liquid = vle.liquid().density;
    println!(
        "      saturation at T = {}: p_sat = {}\n      rho_vapor = {}, rho_liquid = {}",
        temperature, p_sat, rho_vapor, rho_liquid
    );
    assert!(rho_liquid.to_reduced() > rho_vapor.to_reduced());

    // Two reproducible solver chains, spelled out instead of relying on the
    // defaults, because different DFT problems like different iterations:
    //
    // * `solver_log` — Anderson mixing on `ln rho` first, then on `rho`. The
    //   free interface spans three orders of magnitude in density, so iterating
    //   the logarithm first keeps the profile positive.
    // * `solver` — damped Picard iteration to get into the basin of attraction,
    //   then Anderson mixing to converge tightly. This is the robust choice for
    //   the confined profiles, where the wall potential creates a sharp,
    //   liquid-like contact layer.
    //
    // `Verbosity::None` keeps the output deterministic; switch to
    // `Verbosity::Iter` to watch the residual of every iteration.
    let solver_log = DFTSolver::new(Some(Verbosity::None))
        .anderson_mixing(Some(true), Some(50), Some(1e-5), Some(0.15), None)
        .anderson_mixing(Some(false), Some(200), Some(1e-11), Some(0.15), None);
    let solver = DFTSolver::new(Some(Verbosity::None))
        .picard_iteration(None, Some(200), Some(1e-5), Some(0.15))
        .anderson_mixing(None, Some(150), Some(1e-11), None, None);

    // ---------------------------------------------------------------- [3/6] --
    println!("\n[3/6] planar vapour-liquid interface (1-D DFT solve)");
    let interface = PlanarInterface::from_tanh(
        &vle,
        INTERFACE_POINTS,
        from_reduced(INTERFACE_WIDTH),
        critical_temperature,
        false,
    )
    .solve(Some(&solver_log))
    .map_err(profile_error)?;

    let z = interface.profile.z();
    let density = interface.profile.density.to_reduced();
    let profile = density.row(0);
    let n_grid = profile.len();
    let rho_min = profile.iter().copied().fold(f64::INFINITY, f64::min);
    let rho_max = profile.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "      grid: {n_grid} points, z in [{}, {}], volume = {}",
        z.get(0),
        z.get(n_grid - 1),
        interface.profile.volume()
    );
    println!(
        "      density profile: rho(left) = {}, rho(right) = {}, moles = {}",
        interface.profile.density.get((0, 0)),
        interface.profile.density.get((0, n_grid - 1)),
        interface.profile.total_moles()
    );

    // Every DFT density must be a finite, non-negative number.
    assert!(
        profile.iter().all(|rho| rho.is_finite() && *rho >= 0.0),
        "density profile must be finite and non-negative"
    );
    // The tanh initialisation puts liquid on the left, vapour on the right, and
    // both ends must relax onto the bulk coexistence densities.
    assert!(
        profile[0] > profile[n_grid - 1],
        "profile must be monotonic-ish (liquid -> vapour)"
    );
    assert!((rho_max - rho_liquid.to_reduced()).abs() / rho_liquid.to_reduced() < 0.05);
    assert!(rho_min < rho_liquid.to_reduced() * 0.05);

    let surface_tension = interface
        .surface_tension
        .ok_or_else(|| DftError::Profile("surface tension unavailable".into()))?;
    let equimolar_radius = interface
        .equimolar_radius
        .ok_or_else(|| DftError::Profile("equimolar radius unavailable".into()))?;
    println!(
        "      surface tension = {}, equimolar radius = {}",
        surface_tension, equimolar_radius
    );
    assert!(
        surface_tension.to_reduced() > 0.0,
        "surface tension must be positive"
    );

    // ---------------------------------------------------------------- [4/6] --
    println!("\n[4/6] interface diagnostics + solver log");
    let thickness = interface.interfacial_thickness().map_err(profile_error)?;
    let enrichment = interface.interfacial_enrichment();
    println!(
        "      90/10 interface thickness = {}, interfacial enrichment = {:.4}",
        thickness, enrichment[0]
    );
    assert!(
        thickness.to_reduced() > 0.0,
        "interface thickness must be positive"
    );

    if let Some(log) = interface.profile.solver_log.as_ref() {
        let residual = log.residual();
        let iterations = residual.len();
        println!(
            "      solver: {} iteration(s), algorithms {:?}, final residual = {:.3e}",
            iterations,
            log.solver().last().unwrap_or(&"n/a"),
            residual[iterations - 1]
        );
        assert!(residual[iterations - 1].is_finite());
    }

    // ---------------------------------------------------------------- [5/6] --
    println!("\n[5/6] single slit pore under a 9-3 Lennard-Jones wall potential");
    let pore = Pore1D::new(
        Geometry::Cartesian,
        from_reduced(PORE_WIDTH),
        ExternalPotential::LJ93 {
            sigma_ss: 3.0,
            epsilon_k_ss: 100.0,
            rho_s: 0.08,
        },
        Some(PORE_POINTS),
        None,
    );
    println!(
        "      slit width = {}, helium-reference pore volume = {}",
        pore.pore_size,
        pore.pore_volume().map_err(profile_error)?
    );

    let bulk = State::new_npt(
        func,
        temperature,
        p_sat * 0.5,
        (),
        Some(DensityInitialization::Vapor),
    )
    .map_err(functional_error)?;
    let pore_profile = pore
        .initialize(&bulk, None, None)
        .map_err(profile_error)?
        .solve(Some(&solver))
        .map_err(profile_error)?;

    let grand_potential = pore_profile
        .grand_potential
        .ok_or_else(|| DftError::Profile("grand potential unavailable".into()))?;
    let wall_tension = pore_profile
        .interfacial_tension
        .ok_or_else(|| DftError::Profile("interfacial tension unavailable".into()))?;
    let adsorbed = pore_profile.profile.total_moles();
    let enthalpy = pore_profile
        .enthalpy_of_adsorption()
        .map_err(profile_error)?;
    println!(
        "      at p = 0.5 p_sat: adsorbed = {}, grand potential = {}",
        adsorbed, grand_potential
    );
    println!(
        "      wall tension = {}, enthalpy of adsorption = {}",
        wall_tension, enthalpy
    );
    assert!(
        adsorbed.to_reduced() > 0.0,
        "confined fluid must adsorb something"
    );
    assert!(grand_potential.to_reduced().is_finite());

    let pore_density = pore_profile.profile.density.to_reduced();
    assert!(
        pore_density
            .iter()
            .all(|rho| rho.is_finite() && *rho >= 0.0),
        "pore density profile must be finite and non-negative"
    );
    // Confinement + attractive walls enrich the fluid over the bulk vapour.
    let pore_max = pore_density
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "      contact-layer enrichment rho_max / rho_bulk = {:.2}",
        pore_max / bulk.density.to_reduced()
    );
    assert!(pore_max > bulk.density.to_reduced());

    // ---------------------------------------------------------------- [6/6] --
    println!("\n[6/6] adsorption isotherm (loading / mean pore density vs pressure)");
    let pore_volume = pore.pore_volume().map_err(profile_error)?;
    let pressure = RELATIVE_PRESSURES.iter().map(|&x| p_sat * x).collect();
    let isotherm =
        Adsorption1D::adsorption_isotherm(func, temperature, &pressure, &pore, (), Some(&solver))
            .map_err(profile_error)?;

    let isotherm_pressure = isotherm.pressure();
    let loading = isotherm.total_adsorption();
    let mut converged_loadings = Vec::with_capacity(RELATIVE_PRESSURES.len());
    for (i, relative_pressure) in RELATIVE_PRESSURES.iter().enumerate() {
        // `Adsorption::profiles` keeps the per-point `Result`, so a single
        // non-converged pressure never poisons the whole isotherm.
        match &isotherm.profiles[i] {
            Ok(_) => {
                let n_ads = loading.get(i);
                println!(
                    "      p/p_sat = {relative_pressure:.2}: p = {}, n_ads = {}, <rho>_pore = {}",
                    isotherm_pressure.get(i),
                    n_ads,
                    n_ads / pore_volume
                );
                converged_loadings.push(n_ads.to_reduced());
            }
            Err(e) => println!("      p/p_sat = {relative_pressure:.2}: not converged ({e})"),
        }
    }

    assert!(
        converged_loadings.len() >= 2,
        "the isotherm needs at least two converged points to be meaningful"
    );
    assert!(
        converged_loadings
            .iter()
            .all(|n| n.is_finite() && *n >= 0.0),
        "adsorbed amounts must be finite and non-negative"
    );
    // Below capillary condensation the adsorbed film grows with pressure.
    for window in converged_loadings.windows(2) {
        assert!(
            window[1] >= window[0] * (1.0 - 1e-6),
            "adsorption isotherm must not decrease with pressure"
        );
    }
    // The pore always holds more fluid than the same volume of bulk vapour.
    let bulk_reference = (rho_vapor * RELATIVE_PRESSURES[0] * pore_volume).to_reduced();
    assert!(converged_loadings[0] > bulk_reference);

    println!("\nAll classical-DFT checks passed.");
    Ok(())
}
