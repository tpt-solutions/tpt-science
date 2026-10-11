# Examples index

Every crate ships runnable `examples/` programs. This index lists all of them,
with a one-line summary and the first lines of real output from a release-mode run
(captured on Windows). Values are from that run; the last printed digits can differ on
other platforms or toolchains.

Run one with:

```sh
cargo run --release --example <name> -p <crate>
```

CI builds every example and runs each one (the `examples` job in
`.github/workflows/ci.yml`), so a non-zero exit fails the build.

## `tpt-sci-astro`

### `lambert_transfer`

tpt-sci-astro implements the classical Kepler two-body problem: validated

```text
Transfer ellipse: a = 24582.0 km, e = 0.7152
Geometry (state-vector + propagation):
periapsis r =    7000.000 km (expect 7000)
```

### `propagation`

This example is a guided tour of the classical two-body machinery exposed by

```text
round-trip (elements -> state -> elements):
a      7000.0000 ->    7000.0000 km
e       0.010000 ->     0.010000
```

## `tpt-sci-cfd-core`

### `cavity`

A guided tour of the public surface of tpt-sci-cfd-core, exercising the

```text
grid: 32×32 cells, domain 1×1 (dx=0.0312, dy=0.0312)
invalid-grid construction returns: InvalidGrid("cell counts must be > 0")
step   0: max|div(u)| =  0.000e0   KE = 1.6000e1   max|u| = 1.0000
```

### `cfd_showcase`

Showcase of the three v1 capabilities added alongside the explicit

```text
SIMPLE: max velocity divergence after 5 steps = 0.000e0
SST: k finite and positive after decay = true
Unstructured Poisson: max error vs analytic = 5.972e-2
```

### `lid_driven_cavity`

A second, independent example for tpt-sci-cfd-core that exercises the same

```text
after 400 steps: max|div(u)| = 0.000e0   KE = 3.7535e1   max|u| = 1.0000
u along horizontal centerline (x, u, p) at y = 0.510:
x=0.010  u= 0.0000  p= 0.0000
```

## `tpt-sci-climate`

### `energy_balance`

Radiative-forcing and thermal-inertia tour of the tpt-sci-climate surface.

```text
[1] Forcing equivalence: a 2xCO2 warming == a -ΔS solar cut
2xCO2 forcing F = 3.708 W/m^2  ->  compensate with ΔS = -21.19 W/m^2
T_eq(base 1xCO2, S0)        = 288.0638 K
```

### `gcm_multi_band`

Tour of the expanded climate capabilities: multi-band radiative transfer,

```text
tpt-sci-climate: multi-band RT + 3-D tracer + GCM core
multi-band OLR = 262.6 W/m² (effective ε = 0.756)
surface downwelling LW = 167.5 W/m²
```

### `warming`

Tour of the tpt-sci-climate surface — a 0-D global energy-balance model

```text
[1] Baseline: pre-industrial equilibrium & grey-atmosphere check
pre-industrial equilibrium T = 288.064 K  (reference 288 K)
grey layer eps = 0.7800  ->  eps_eff = 1 - eps/2 = 0.61
```

## `tpt-sci-dft-classical`

### `adsorption`

tpt-sci-dft-classical wraps the feos framework and adds a

```text
[1/6] parameters + Helmholtz energy functional
../../parameters/pcsaft/esper2023.json unavailable (functional error: The system cannot find the path specifie
PC-SAFT propane: components = 1, m = 2.0018, sigma = 3.6184 A, eps/k = 208.11 K
```

### `isotherm`

tpt-sci-dft-classical wraps the feos framework and adds a

```text
[1/3] parameters + Helmholtz energy functional
PC-SAFT propane: components = 1, m = 2.0018, sigma = 3.6184 A, eps/k = 208.11 K
[2/3] bulk reference states across a temperature sweep
```

### `square_gradient`

From-scratch square-gradient classical DFT demo.

```text
target bulk density n_b = 0.1, analytic bulk = 0.100000
1-D hard wall: wall density = 0.0240, far-field density = 0.1000, surface tension gamma = 0.01540, iters = 315
3-D core: far-field density = 0.0965 (1-D bulk = 0.1000), excess free energy = 0.03627, iters = 98
```

## `tpt-sci-dft-electronic`

### `harmonic_atom`

A from-scratch electronic-structure DFT demo exercising the public surface of

```text
grid: n = 101 points, dx = 0.1200, x in [-6.0, 6.0]
grid error path OK: InvalidGrid raised
LDA exchange-correlation energy density e_xc(ρ):
```

### `hydrogen_atom`

A second, complementary tour of the crate's 1-D Kohn–Sham LDA surface, this

```text
grid error path OK: InvalidGrid raised
setup error path OK: InvalidSetup (0 e⁻)
LDA e_xc(0.5) = -0.5904 Ha (attractive)
```

## `tpt-sci-electrophys`

### `ap_wave`

A guided demo of the crate's public surface, built on tpt_sci_ode (membrane

```text
resting V =  -65.00 mV  I_ion =    0.003 µA/cm²  gating m=0.053 h=0.596 n=0.318
peak V    =   43.48 mV at t = 306.00 ms  (spike amplitude = 108.48 mV)
after  1200 ms: V =    8.97 mV  m=0.992 h=0.180 n=0.734 (recovery)
```

### `bidomain_demo`

Demonstrate the **bidomain** capability of tpt-sci-electrophys: a 2-D tissue

```text
step   0: max Vm =    39.38 mV, Ve ∈ [  -50.31,     2.19] mV
step  80: max Vm =    40.84 mV, Ve ∈ [  -50.39,     2.59] mV
step 160: max Vm =    26.66 mV, Ve ∈ [  -42.83,     3.10] mV
```

### `cable_propagation`

A second, complementary tour of the crate's public surface, focused on

```text
sub-threshold bias I=0     : spikes = 0, V end =  -65.00 mV
supra-threshold bias I=10  : spikes = 7, V end =  -62.28 mV
threshold confirmed: 0 µA/cm² is silent, 10 µA/cm² drives 7 spikes
```

## `tpt-sci-grid`

### `diffusion`

tpt-sci-grid builds discrete PDE operators on

```text
1-D Gaussian diffusion (2000 explicit steps):
Dirichlet: peak 0.3334  (mass 12.533 -> 12.511, leaks out)
Neumann:   peak 0.3334  (mass 12.533 -> 12.536, conserved)
```

### `heat_2d`

tpt-sci-grid assembles discrete PDE operators on structured tensor-product

```text
grid: 41x41 nodes over [0,1]x[0,1], dx = dy = 0.0250, D = 0.01, dt = 0.001, steps = 1000 (t_final = 1)
[Dirichlet]  u(x,y,0) = sin(πx)·sin(πy),  u = 0 on the boundary
centre field u(0.500,0.500,1.0) = 0.820936
```

## `tpt-sci-hemodynamics`

### `arterial_segment`

A guided demonstration of the tpt-sci-hemodynamics public surface: the

```text
1. Tube-law stiffness
β (soft E=1e4)  = 28.025
β (stiff E=1e6) = 280.250
```

### `wave_reflection`

A second, complementary tour of the crate's 1-D compliant-vessel surface,

```text
1. Characteristic impedance Z0 = ρ·c / A  and reflection R
parent  (A0=1, stiff): Z0 = 10.248
matched daughter        : R = 0.000
```

### `womersley_coupling`

Demonstrates the two v1 features closed out for tpt-sci-hemodynamics:

```text
1. Womersley velocity amplitude |u(r)| (r in [0, R])
r/R=0.0: |u|=0.1503
r/R=0.5: |u|=0.1509
```

## `tpt-sci-image`

### `radon_sinogram`

This example complements reconstruction.rs.

```text
Shepp–Logan phantom (96x96) forward projection:
sinogram            : 120 angles x 96 detector bins
total projected mass : 3874.3 (rows of sinogram)
```

### `reconstruction`

Parallel-beam computed tomography recovers a cross-section from its

```text
2-D (64x64) two-disc phantom reconstruction:
sinogram            : 90x64
FBP   reconstruction RMSE : 0.0684
```

## `tpt-sci-kinetics`

### `langmuir_hinshelwood`

A second, complementary tour of the tpt_sci_kinetics public surface — the

```text
1) Langmuir isotherm  θ_A(p) = K·p/(1+K·p)   (K = 2)
p        θ_A     closed      1-θ_A
0.010     0.0196     0.0196     0.9804
```

### `reactor`

A guided tour of the tpt_sci_kinetics public surface, and of how it

```text
1) Arrhenius surface step  (A = 1e2, Ea = 50000 J/mol)
k(600 K) = 4.438e-3
k(900 K) = 1.254e-1
```

## `tpt-sci-md`

### `lennard_jones`

A small, fully deterministic demo that exercises a broad slice of the

```text
r= 0.900  F_x= -138.6596  U=    6.6524
r= 1.000  F_x=  -24.0000  U=    0.0163
r= 1.122  F_x=   -0.0265  U=   -0.9837
```

### `radial_distribution`

A second, structurally-focused tour of the tpt_sci_md public surface — the

```text
eq    0  T=  0.0405  E_kin=    3.8267
eq  300  T=  0.6567  E_kin=   62.0567
eq  600  T=  0.6453  E_kin=   60.9818
```

## `tpt-sci-ocean`

### `ocean3d`

Tour of the 3-D ocean core and data assimilation in tpt-sci-ocean.

```text
tpt-sci-ocean: 3-D core + assimilation tour
1. Warm anomaly: u (left of centre, bottom) 0.0000e0 -> 1.0066e-5 m/s (flow develops toward the anomaly).
2. Non-hydrostatic projection: max interior divergence 8.000e-4 -> 9.432e-13.
```

### `shallow_water`

Ocean demo: a tour of the 2-D shallow-water surface exposed by

```text
tpt-sci-ocean ShallowWater tour on a 48x48 basin (g=9.81, f=1e-4, dt=0.0005).
step   0: max_speed=0.0000 m/s | h∈[10.0000,10.4000] mean=10.0109 | E_kin=0.000e0 E_pot=1.070e-2
step  20: max_speed=0.0827 m/s | h∈[10.0000,10.0724] mean=10.0109 | E_kin=3.162e-3 E_pot=2.396e-3
```

### `shallow_water_wave`

Ocean demo: a **dam-break problem** in the 2-D shallow-water

```text
tpt-sci-ocean dam-break (Riemann) on a 64x64 basin, f = 0.
g=  9.81 step   0: front_x = 0.7266 m | max_speed = 0.0000 m/s | E_kin = 0.000e0 E_pot = 3.863e1
g=  9.81 step  50: front_x = 0.9375 m | max_speed = 1.6593 m/s | E_kin = 6.998e0 E_pot = 2.769e1
```

## `tpt-sci-ode`

### `predator_prey`

first integral.

```text
x' = α x - β x y,   y' = δ x y - γ y
α = 0.6666666666666666, β = 1.3333333333333333, δ = 1, γ = 1,  (x0, y0) = (1, 1)
equilibrium: x* = 1, y* = 0.5
```

### `van_der_pol`

Van der Pol oscillator — a guided tour of the tpt-sci-ode public API.

```text
y0' = y1,   y1' = mu (1 - y0^2) y1 - y0
mu = 1, y(0) = [2.0, 0.0], integrating to t = 30
[1] Constructors: OdeProblem::new vs OdeProblemBuilder
```

## `tpt-sci-physics-rigid`

### `collision`

A runnable tour of the tpt-sci-physics-rigid public surface.

```text
free fall: y 4.000 -> 2.775 (Δy = -1.225, expected -1.225)
wall bounce: vx +1.000 -> 1.000 (reflected, restitution = 1.0)
duplicate-id rejected as duplicate body id 0
```

### `pendulum`

A second, self-contained runnable example — distinct from collision.rs.

```text
initial release angle θ0 = 0.500 rad
initial COM = [0.959, -1.755], q = [0.969, 0.000, 0.000, 0.247]
t= 0.001s  θ=+0.5000 rad  ω=-0.0023 rad/s  E= 2.399371 J
```

## `tpt-sci-ppl`

### `logistic_regression`

This example complements posterior.rs, which fits *continuous* (Gaussian /

```text
Recovered parameters (4 chains x 2000 draws):
a (intercept) = +0.600   (truth +1.0,  rhat = 1.0005, ess = 6083)
b (slope)     = +1.764   (truth +2.0,  rhat = 1.0006, ess = 6581)
```

### `posterior`

This example exercises a broad slice of the probabilistic-programming

```text
Linear regression (default 2-chain fit):
slope    = 1.874  (truth 2.0,  ess = 1374)
intercept= 0.995  (truth 1.0,  ess = 1535)
```

## `tpt-sci-quantum`

### `bell_ghz`

This example is a guided walk through the public surface of the

```text
Bell |Φ+> = (|00> + |11>)/√2
P(|00>) = 0.500, P(|11>) = 0.500
<Z⊗Z> = 1.000 (expect +1)
```

### `grover_oracle`

This example explores a genuinely different quantum primitive from the

```text
Phase oracle on H⊗H|00> (all amplitudes start at +1/2):
|00> amplitude = +0.5000 (expect +0.5000)
|01> amplitude = -0.5000 (expect -0.5000)
```

## `tpt-sci-reaction-network`

### `gene_expression`

The SIR example built an epidemic compartmental model.

```text
programmatic builder and textual DSL produce identical IR
species : ["G", "Gstar", "M", "P"]
params  : ["k_on", "k_off", "k_t", "k_dm", "k_tl", "k_dp"]
```

### `sir`

The SIR model splits a closed population N into Susceptible ,

```text
programmatic builder and textual DSL produce identical IR
species : ["S", "I", "R"]
params  : ["beta", "gamma"]
```

## `tpt-sci-sim-core`

### `decay_coupled`

Heterogeneous sub-models stepping to a shared target time, with checkpoint

```text
[1] Sub-model anatomy (used bare via the SubModel trait)
OdeSubModel id = decay, time = 0, max_step = inf
method after set_method(Tsit45) = Tsit45
```

### `multi_scale_cookbook`

Multi-scale cookbook: a reaction network built with the

```text
SIR species: ["S", "I", "R"], stoichiometry (S,I,R rows) = [[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]]
initial reaction rates r = [19.8000, 4.0000] (beta*S*I, gamma*I)
[1] Couplings (source = sir, target = pathogen)
```
