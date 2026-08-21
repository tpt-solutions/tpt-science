use std::collections::HashMap;
use std::sync::Arc;

use crate::error::ReactionNetworkError;
use crate::ssa::SsaTrajectory;
use crate::tau_leap::{TauLeapConfig, sample_poisson, sample_standard_normal};

/// A user-supplied reaction rate law `r(y, p)` over the current state and the
/// parameter vector.
pub(crate) type RateFn = Arc<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>;

/// A reactant or product term: a species (by index) together with its
/// stoichiometric coefficient in a single reaction.
///
/// For `A + 2 B --> C`, `A` contributes a term `(a_index, 1.0)` and `B` a term
/// `(b_index, 2.0)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    /// Index of the species in the network (as returned by
    /// [`ReactionNetwork::species`]).
    pub species: usize,
    /// Stoichiometric coefficient of this species in the term.
    pub stoichiometry: f64,
}

/// The kinetic law governing a single reaction's rate.
#[derive(Clone)]
pub enum RateLaw {
    /// Classic mass-action kinetics: `rate = k · Π_i y_i^{ν_i}`, where `k` is a
    /// named rate constant and `ν_i` are the reactant stoichiometric
    /// exponents. This is the default for the textual DSL.
    MassAction {
        /// Name of the rate-constant parameter (must be declared on the
        /// network, e.g. via [`ReactionNetwork::parameter`]).
        k: String,
    },
    /// A user-supplied rate expression `r(y, p)` depending on the current
    /// state `y` and the parameter vector `p`. Use this for Michaelis–Menten,
    /// Hill, or any other non-mass-action kinetics.
    Custom(RateFn),
}

impl std::fmt::Debug for RateLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLaw::MassAction { k } => f.debug_struct("MassAction").field("k", k).finish(),
            RateLaw::Custom(_) => f.write_str("Custom(<closure>)"),
        }
    }
}

impl RateLaw {
    /// Convenience constructor for a mass-action rate with the named constant.
    #[must_use]
    pub fn mass_action(name: &str) -> Self {
        RateLaw::MassAction {
            k: name.to_string(),
        }
    }

    /// Convenience constructor for a custom user-supplied rate law.
    #[must_use]
    pub fn custom<F>(f: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static,
    {
        RateLaw::Custom(Arc::new(f))
    }
}

/// A single chemical reaction: reactants `-->` products under a [`RateLaw`].
#[derive(Debug, Clone)]
pub struct Reaction {
    /// Reactant terms (consumed species and their stoichiometry).
    pub reactants: Vec<Term>,
    /// Product terms (produced species and their stoichiometry).
    pub products: Vec<Term>,
    /// Kinetic law for the reaction rate.
    pub rate: RateLaw,
}

/// Builder for a reaction network. Species and parameters are registered by
/// name (registered-on-first-use), and reactions are appended with explicit
/// species indices. Call [`ReactionNetwork::build`] to compile into a
/// [`ReactionSystem`].
///
/// The textual [`ReactionNetwork::from_dsl`] helper constructs one of these
/// internally.
#[derive(Debug, Clone, Default)]
pub struct ReactionNetwork {
    species: Vec<String>,
    species_index: HashMap<String, usize>,
    params: Vec<(String, f64)>,
    param_index: HashMap<String, usize>,
    reactions: Vec<Reaction>,
}

impl ReactionNetwork {
    /// Create an empty network.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a species by name, returning its index. If the name is already
    /// known, the existing index is returned (idempotent).
    pub fn species(&mut self, name: &str) -> usize {
        if let Some(&i) = self.species_index.get(name) {
            return i;
        }
        let i = self.species.len();
        self.species.push(name.to_string());
        self.species_index.insert(name.to_string(), i);
        i
    }

    /// Declare (or overwrite) a parameter `name` with `value`, returning its
    /// index.
    pub fn parameter(&mut self, name: &str, value: f64) -> usize {
        if let Some(&i) = self.param_index.get(name) {
            self.params[i].1 = value;
            return i;
        }
        let i = self.params.len();
        self.params.push((name.to_string(), value));
        self.param_index.insert(name.to_string(), i);
        i
    }

    /// Append a reaction. `reactants`/`products` are `(species_index,
    /// stoichiometry)` pairs.
    pub fn reaction(
        &mut self,
        reactants: &[(usize, f64)],
        products: &[(usize, f64)],
        rate: RateLaw,
    ) -> &mut Self {
        let r = Reaction {
            reactants: reactants
                .iter()
                .map(|&(species, stoichiometry)| Term {
                    species,
                    stoichiometry,
                })
                .collect(),
            products: products
                .iter()
                .map(|&(species, stoichiometry)| Term {
                    species,
                    stoichiometry,
                })
                .collect(),
            rate,
        };
        self.reactions.push(r);
        self
    }

    /// Parse a Catalyst.jl-style textual model and append its reactions to this
    /// network (species and rate constants are registered as encountered).
    ///
    /// # Errors
    ///
    /// Returns [`ReactionNetworkError::Dsl`] if a line cannot be parsed.
    pub fn parse_dsl(&mut self, src: &str) -> Result<(), ReactionNetworkError> {
        for raw in src.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (k, body) = line.split_once(',').ok_or_else(|| {
                ReactionNetworkError::Dsl(format!("missing rate constant: {line}"))
            })?;
            let k = k.trim();
            if k.is_empty() {
                return Err(ReactionNetworkError::Dsl(format!(
                    "empty rate constant in: {line}"
                )));
            }
            // Ensure the rate constant exists (value 0.0; caller sets it later).
            if !self.param_index.contains_key(k) {
                self.parameter(k, 0.0);
            }
            let body = body.trim();
            let sep = body
                .find("-->")
                .or_else(|| body.find("=>"))
                .or_else(|| body.find("->"));
            let sep = sep.ok_or_else(|| {
                ReactionNetworkError::Dsl(format!("missing '-->' / '=>' / '->' separator: {body}"))
            })?;
            // Disambiguate '->' from a negative coefficient like '-2 A' — a
            // genuine separator is followed by '>' or is '->'/'-->'/'=>'.
            // `-->` is the 3-char arrow; `=>` and `->` are both 2-char arrows.
            let arrow_len = if body[sep..].starts_with("-->") { 3 } else { 2 };
            let lhs = body[..sep].trim();
            let rhs = &body[sep + arrow_len..];
            let rhs = rhs.trim();
            let reactants = parse_side(self, lhs)?;
            let products = parse_side(self, rhs)?;
            self.reactions.push(Reaction {
                reactants,
                products,
                rate: RateLaw::mass_action(k),
            });
        }
        Ok(())
    }

    /// Build a complete [`ReactionSystem`] from the registered species,
    /// parameters, and reactions.
    ///
    /// # Errors
    ///
    /// Returns an error if any reaction is empty, or if a `MassAction` rate
    /// references an undeclared rate constant.
    pub fn build(&self) -> Result<ReactionSystem, ReactionNetworkError> {
        let n_species = self.species.len();
        let mut reactions = Vec::with_capacity(self.reactions.len());
        for r in &self.reactions {
            if r.reactants.is_empty() && r.products.is_empty() {
                return Err(ReactionNetworkError::EmptyReaction);
            }
            let rate = match &r.rate {
                RateLaw::MassAction { k } => {
                    let idx =
                        self.param_index.get(k).copied().ok_or_else(|| {
                            ReactionNetworkError::UndefinedRateConstant(k.clone())
                        })?;
                    CompiledRate::MassAction { k: idx }
                }
                RateLaw::Custom(f) => CompiledRate::Custom(f.clone()),
            };
            let mut stoich_col = vec![0.0; n_species];
            for t in &r.reactants {
                stoich_col[t.species] -= t.stoichiometry;
            }
            for t in &r.products {
                stoich_col[t.species] += t.stoichiometry;
            }
            let reactant_powers: Vec<(usize, f64)> = r
                .reactants
                .iter()
                .map(|t| (t.species, t.stoichiometry))
                .collect();
            reactions.push(CompiledReaction {
                reactant_powers,
                stoich_col,
                rate,
            });
        }
        let params: Vec<f64> = self.params.iter().map(|(_, v)| *v).collect();
        let param_names: Vec<String> = self.params.iter().map(|(n, _)| n.clone()).collect();
        Ok(ReactionSystem {
            species: self.species.clone(),
            species_index: self.species_index.clone(),
            params,
            param_names,
            param_index: self.param_index.clone(),
            reactions,
        })
    }

    /// Build a [`ReactionSystem`] directly from a Catalyst.jl-style textual
    /// model.
    ///
    /// # Errors
    ///
    /// Returns an error if the DSL cannot be parsed or the network cannot be
    /// built (see [`ReactionNetwork::parse_dsl`] / [`ReactionNetwork::build`]).
    pub fn from_dsl(src: &str) -> Result<ReactionSystem, ReactionNetworkError> {
        let mut net = ReactionNetwork::new();
        net.parse_dsl(src)?;
        net.build()
    }
}

/// A compiled, immutable reaction network ready for evaluation / integration.
#[derive(Debug, Clone)]
pub struct ReactionSystem {
    species: Vec<String>,
    species_index: HashMap<String, usize>,
    params: Vec<f64>,
    param_names: Vec<String>,
    param_index: HashMap<String, usize>,
    reactions: Vec<CompiledReaction>,
}

#[derive(Debug, Clone)]
struct CompiledReaction {
    /// `(species_index, reactant_stoichiometry)` pairs used to compute the
    /// mass-action propensity `Π y_i^{ν_i}`. Empty for `Custom` rates.
    reactant_powers: Vec<(usize, f64)>,
    /// Net change of each species per firing of this reaction (length =
    /// number of species). This is column `j` of the stoichiometry matrix.
    stoich_col: Vec<f64>,
    rate: CompiledRate,
}

#[derive(Clone)]
enum CompiledRate {
    MassAction { k: usize },
    Custom(RateFn),
}

impl std::fmt::Debug for CompiledRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompiledRate::MassAction { k } => f.debug_struct("MassAction").field("k", k).finish(),
            CompiledRate::Custom(_) => f.write_str("Custom(<closure>)"),
        }
    }
}

impl ReactionSystem {
    /// Number of species in the network.
    #[must_use]
    pub fn n_species(&self) -> usize {
        self.species.len()
    }

    /// Number of reactions in the network.
    #[must_use]
    pub fn n_reactions(&self) -> usize {
        self.reactions.len()
    }

    /// Species names, in index order.
    #[must_use]
    pub fn species_names(&self) -> &[String] {
        &self.species
    }

    /// Look up a species index by name.
    #[must_use]
    pub fn species_index(&self, name: &str) -> Option<usize> {
        self.species_index.get(name).copied()
    }

    /// Get a parameter value by name.
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<f64> {
        self.param_index.get(name).map(|&i| self.params[i])
    }

    /// Set a parameter value by name (e.g. after constructing from the DSL,
    /// whose rate constants default to `0.0`).
    ///
    /// # Errors
    ///
    /// Returns [`ReactionNetworkError::UnknownParameter`] if the name was never
    /// declared.
    pub fn set_parameter(&mut self, name: &str, value: f64) -> Result<(), ReactionNetworkError> {
        let i = self
            .param_index
            .get(name)
            .copied()
            .ok_or_else(|| ReactionNetworkError::UnknownParameter(name.to_string()))?;
        self.params[i] = value;
        Ok(())
    }

    /// Build an initial state vector from `(species_name, value)` pairs. Species
    /// not mentioned default to `0.0`.
    ///
    /// # Errors
    ///
    /// Returns [`ReactionNetworkError::UnknownSpecies`] if a named species was
    /// never declared.
    pub fn initial_state(&self, values: &[(&str, f64)]) -> Result<Vec<f64>, ReactionNetworkError> {
        let mut y = vec![0.0; self.species.len()];
        for &(name, v) in values {
            let i = self
                .species_index
                .get(name)
                .copied()
                .ok_or_else(|| ReactionNetworkError::UnknownSpecies(name.to_string()))?;
            y[i] = v;
        }
        Ok(y)
    }

    /// Stoichiometry matrix `S`, indexed `S[species][reaction]` = net change of
    /// that species when the reaction fires once. Hence `dy/dt = S · r(y)`.
    #[must_use]
    pub fn stoichiometry_matrix(&self) -> Vec<Vec<f64>> {
        let n_reactions = self.reactions.len();
        let mut s: Vec<Vec<f64>> = (0..self.species.len())
            .map(|_| vec![0.0; n_reactions])
            .collect();
        for (j, r) in self.reactions.iter().enumerate() {
            for (i, col) in s.iter_mut().enumerate() {
                col[j] = r.stoich_col[i];
            }
        }
        s
    }

    /// Evaluate the per-reaction rate vector `r(y)` at state `y`.
    #[must_use]
    pub fn reaction_rates(&self, y: &[f64]) -> Vec<f64> {
        self.reactions
            .iter()
            .map(|r| match &r.rate {
                CompiledRate::MassAction { k } => {
                    let mut v = self.params[*k];
                    for &(sp, pow) in &r.reactant_powers {
                        v *= y[sp].powf(pow);
                    }
                    v
                }
                CompiledRate::Custom(f) => f(y, &self.params),
            })
            .collect()
    }

    /// Per-reaction stochastic propensities `a_j(y)` at state `y`.
    ///
    /// For [`CompiledRate::MassAction`] the propensity is the combinatorial
    /// mass-action term `k · Π_i (y_i)_ν_i` where `(y_i)_ν_i` is the falling
    /// factorial (so it correctly handles integer reactant counts, e.g. a
    /// dimerisation `A + A → B` gets rate `k·A·(A−1)`). For non-integer
    /// stoichiometry the deterministic `y_i^ν_i` is used. For
    /// [`CompiledRate::Custom`] rates the same differentiable rate expression is
    /// reused as the propensity (a deterministic approximation).
    fn propensities(&self, y: &[f64]) -> Vec<f64> {
        self.reactions
            .iter()
            .map(|r| match &r.rate {
                CompiledRate::MassAction { k } => {
                    let mut v = self.params[*k];
                    for &(sp, nu) in &r.reactant_powers {
                        v *= falling_factorial(y[sp], nu);
                    }
                    v
                }
                CompiledRate::Custom(f) => f(y, &self.params),
            })
            .collect()
    }

    /// Draw a stochastic trajectory of this network via Gillespie's direct
    /// method (the exact SSA for the underlying continuous-time Markov chain).
    ///
    /// `y0` gives the initial (finite, non-negative) species counts and the
    /// chain is advanced until `t_max`. `rng` must return uniform variates in
    /// `[0, 1)`. The returned [`SsaTrajectory`] records the initial state, every
    /// post-reaction state, and a final state clamped at `t_max`.
    ///
    /// # Errors
    ///
    /// Returns [`ReactionNetworkError::StateDimension`] if `y0` does not have one
    /// entry per species, or [`ReactionNetworkError::InvalidState`] if any
    /// initial count is non-finite or negative.
    pub fn simulate_ssa<F: FnMut() -> f64>(
        &self,
        y0: &[f64],
        t_max: f64,
        rng: &mut F,
    ) -> Result<SsaTrajectory, ReactionNetworkError> {
        self.validate_stochastic_state(y0)?;
        let mut state = y0.to_vec();
        let mut t = 0.0_f64;
        let mut times = vec![0.0];
        let mut states = vec![state.clone()];
        loop {
            let a = self.propensities(&state);
            let a0: f64 = a.iter().sum();
            if a0 <= 0.0 {
                break;
            }
            let r1 = (rng)();
            if r1 <= 0.0 {
                break;
            }
            let dt = -(1.0 / a0) * r1.ln();
            let t_next = t + dt;
            if t_next > t_max {
                break;
            }
            t = t_next;
            let r2 = (rng)() * a0;
            let mut cum = 0.0_f64;
            let mut chosen = 0usize;
            for (j, &aj) in a.iter().enumerate() {
                cum += aj;
                if r2 < cum {
                    chosen = j;
                    break;
                }
            }
            for (s, dc) in state
                .iter_mut()
                .zip(self.reactions[chosen].stoich_col.iter())
            {
                *s += *dc;
            }
            times.push(t);
            states.push(state.clone());
        }
        times.push(t_max);
        states.push(state);
        Ok(SsaTrajectory { times, states })
    }

    /// Validate a stochastic simulation's initial state: it must have one
    /// entry per species, and every entry must be finite and non-negative.
    /// Shared by [`Self::simulate_ssa`], [`Self::simulate_tau_leaping`], and
    /// [`Self::simulate_cle`].
    fn validate_stochastic_state(&self, y0: &[f64]) -> Result<(), ReactionNetworkError> {
        if y0.len() != self.n_species() {
            return Err(ReactionNetworkError::StateDimension {
                got: y0.len(),
                expected: self.n_species(),
            });
        }
        for (i, &v) in y0.iter().enumerate() {
            if !v.is_finite() || v < 0.0 {
                return Err(ReactionNetworkError::InvalidState(format!(
                    "species {i} count {v} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    /// Simplified adaptive tau-selection: bound the expected drift and
    /// standard deviation of every species' propensity-driven change per
    /// step to `epsilon` of its current population (a simplified version of
    /// the Cao–Gillespie–Petzold rule — see [`TauLeapConfig::epsilon`] for
    /// what is simplified away).
    fn select_tau(&self, y: &[f64], a: &[f64], epsilon: f64) -> f64 {
        let n = self.species.len();
        let mut mu = vec![0.0_f64; n];
        let mut sigma2 = vec![0.0_f64; n];
        for (j, r) in self.reactions.iter().enumerate() {
            let aj = a[j];
            if aj <= 0.0 {
                continue;
            }
            for (i, &dc) in r.stoich_col.iter().enumerate() {
                if dc == 0.0 {
                    continue;
                }
                mu[i] += dc * aj;
                sigma2[i] += dc * dc * aj;
            }
        }
        let mut tau = f64::INFINITY;
        for i in 0..n {
            let bound = (epsilon * y[i]).max(1.0);
            if mu[i].abs() > 0.0 {
                tau = tau.min(bound / mu[i].abs());
            }
            if sigma2[i] > 0.0 {
                tau = tau.min(bound * bound / sigma2[i]);
            }
        }
        if !tau.is_finite() || tau <= 0.0 {
            // No species has any propensity-driven change under this bound
            // (should not normally happen once a0 > 0 has been checked by
            // the caller) — fall back to a conservatively small step.
            tau = 1e-6;
        }
        tau
    }

    /// Draw an approximate stochastic trajectory via explicit tau-leaping
    /// (Gillespie 2001): repeatedly advance the state by a step `τ`, firing
    /// each reaction `k_j ~ Poisson(a_j(y)·τ)` times, independently across
    /// reactions. This trades exactness for speed relative to
    /// [`Self::simulate_ssa`] — valid whenever `τ` is small enough that
    /// propensities change little over the step (the *leap condition*),
    /// which [`TauLeapConfig`]'s adaptive mode enforces automatically unless
    /// `fixed_tau` is set. See [`crate::tau_leap`] for background.
    ///
    /// `y0` gives the initial (finite, non-negative) species counts and the
    /// trajectory is advanced until `t_max`. `rng` must return uniform
    /// variates in `[0, 1)`.
    ///
    /// # Errors
    ///
    /// Returns [`ReactionNetworkError::StateDimension`] if `y0` does not have
    /// one entry per species, or [`ReactionNetworkError::InvalidState`] if any
    /// initial count is non-finite or negative.
    pub fn simulate_tau_leaping<F: FnMut() -> f64>(
        &self,
        y0: &[f64],
        t_max: f64,
        config: &TauLeapConfig,
        rng: &mut F,
    ) -> Result<SsaTrajectory, ReactionNetworkError> {
        self.validate_stochastic_state(y0)?;
        let mut state = y0.to_vec();
        let mut t = 0.0_f64;
        let mut times = vec![0.0];
        let mut states = vec![state.clone()];
        while t < t_max {
            let a = self.propensities(&state);
            let a0: f64 = a.iter().sum();
            if a0 <= 0.0 {
                break;
            }
            let mut tau = match config.fixed_tau {
                Some(tau) => tau,
                None => self.select_tau(&state, &a, config.epsilon),
            };
            tau = tau.min(t_max - t);
            if tau <= 0.0 {
                break;
            }
            // Fire each reaction a Poisson(a_j * tau) number of times; if
            // the leap would drive any species negative (an artifact of the
            // Poisson approximation — impossible in the exact SSA), halve
            // tau and retry before falling back to clamping at zero.
            let mut attempt = 0;
            loop {
                let mut next = state.clone();
                for (j, &aj) in a.iter().enumerate() {
                    if aj <= 0.0 {
                        continue;
                    }
                    let k = sample_poisson(aj * tau, rng);
                    if k == 0.0 {
                        continue;
                    }
                    for (s, dc) in next.iter_mut().zip(self.reactions[j].stoich_col.iter()) {
                        *s += dc * k;
                    }
                }
                let ok = next.iter().all(|&v| v >= 0.0);
                if ok || attempt >= config.max_step_halvings {
                    for v in next.iter_mut() {
                        if *v < 0.0 {
                            *v = 0.0;
                        }
                    }
                    state = next;
                    break;
                }
                tau *= 0.5;
                attempt += 1;
            }
            t += tau;
            let t_clamped = t.min(t_max);
            times.push(t_clamped);
            states.push(state.clone());
            t = t_clamped;
        }
        if times.last() != Some(&t_max) {
            times.push(t_max);
            states.push(state);
        }
        Ok(SsaTrajectory { times, states })
    }

    /// Draw an approximate trajectory via the Chemical Langevin Equation
    /// (CLE), Gillespie's diffusion approximation to the same jump process,
    /// integrated by explicit Euler–Maruyama with fixed step `dt`:
    ///
    /// ```text
    /// dY = S·a(Y) dt + S·diag(√a(Y)) dW
    /// ```
    ///
    /// where `dW` is a vector of independent `N(0, dt)` increments, one per
    /// reaction. This is only sensible in the regime where propensities stay
    /// large enough that species counts can be treated as continuous, so it
    /// is typically less robust than [`Self::simulate_tau_leaping`] at small
    /// populations; negative excursions (again an artifact of the Gaussian
    /// approximation) are clamped at zero. See [`crate::tau_leap`] for
    /// background.
    ///
    /// # Errors
    ///
    /// Returns [`ReactionNetworkError::StateDimension`] if `y0` does not have
    /// one entry per species, or [`ReactionNetworkError::InvalidState`] if any
    /// initial count is non-finite or negative.
    pub fn simulate_cle<F: FnMut() -> f64>(
        &self,
        y0: &[f64],
        t_max: f64,
        dt: f64,
        rng: &mut F,
    ) -> Result<SsaTrajectory, ReactionNetworkError> {
        self.validate_stochastic_state(y0)?;
        let mut state = y0.to_vec();
        let mut t = 0.0_f64;
        let mut times = vec![0.0];
        let mut states = vec![state.clone()];
        while t < t_max {
            let step = dt.min(t_max - t);
            if step <= 0.0 {
                break;
            }
            let a = self.propensities(&state);
            let mut next = state.clone();
            for (j, &aj) in a.iter().enumerate() {
                if aj <= 0.0 {
                    continue;
                }
                let drift = aj * step;
                let noise = (aj * step).sqrt() * sample_standard_normal(rng);
                let dxj = drift + noise;
                for (s, dc) in next.iter_mut().zip(self.reactions[j].stoich_col.iter()) {
                    *s += dc * dxj;
                }
            }
            for v in next.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            state = next;
            t += step;
            times.push(t);
            states.push(state.clone());
        }
        Ok(SsaTrajectory { times, states })
    }

    /// Evaluate the ODE right-hand side `dy/dt = S · r(y)` at state `y`,
    /// writing the result into `dydt`. `t` is ignored (CRN kinetics are
    /// autonomous); supply a custom rate law if time dependence is required.
    pub fn eval_rhs(&self, y: &[f64], dydt: &mut [f64]) {
        debug_assert_eq!(dydt.len(), self.species.len());
        for (i, slot) in dydt.iter_mut().enumerate() {
            *slot = 0.0;
            for r in &self.reactions {
                let rate = match &r.rate {
                    CompiledRate::MassAction { k } => {
                        let mut v = self.params[*k];
                        for &(sp, pow) in &r.reactant_powers {
                            v *= y[sp].powf(pow);
                        }
                        v
                    }
                    CompiledRate::Custom(f) => f(y, &self.params),
                };
                *slot += r.stoich_col[i] * rate;
            }
        }
    }

    /// Parameter names, in the canonical (index) order used by custom rate
    /// laws, which receive the parameter vector in this order.
    #[must_use]
    pub fn parameter_names(&self) -> &[String] {
        &self.param_names
    }
}

/// Falling factorial `(y)_ν = y·(y−1)·…·(y−ν+1)` for a (typically integer)
/// stoichiometric order `ν`. Returns `1.0` for `ν == 0`, `0.0` when `y < ν`,
/// and `y^ν` for non-integer `ν` (so it degrades gracefully to the deterministic
/// mass-action power when counts are fractional).
fn falling_factorial(y: f64, nu: f64) -> f64 {
    if nu == 0.0 {
        return 1.0;
    }
    let m = nu as usize;
    if (m as f64 - nu).abs() < 1e-12 {
        if y < m as f64 {
            return 0.0;
        }
        let mut prod = 1.0_f64;
        for p in 0..m {
            prod *= y - p as f64;
        }
        prod
    } else {
        y.powf(nu)
    }
}

/// Parse one side of a reaction (a `+`-separated list of optional-coefficient
/// species names) into terms, registering any new species on the network.
fn parse_side(net: &mut ReactionNetwork, s: &str) -> Result<Vec<Term>, ReactionNetworkError> {
    let s = s.trim();
    if s.is_empty() || s == "0" || s == "∅" {
        return Ok(Vec::new());
    }
    let mut terms = Vec::new();
    for part in s.split('+') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (coef, name) = parse_term(part);
        let idx = net.species(name);
        terms.push(Term {
            species: idx,
            stoichiometry: coef,
        });
    }
    Ok(terms)
}

/// Split a term like `2 A` into `(2.0, "A")`, defaulting the coefficient to
/// `1.0` when none is written.
fn parse_term(part: &str) -> (f64, &str) {
    let digits = part
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .map(|c| c.len_utf8())
        .sum::<usize>();
    if digits == 0 {
        (1.0, part)
    } else {
        let n: f64 = part[..digits].parse().unwrap_or(1.0);
        let rest = part[digits..].trim();
        (n, rest)
    }
}
