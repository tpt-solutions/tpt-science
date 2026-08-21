use thiserror::Error;

/// Errors raised while building, compiling, or integrating a
/// [`ReactionSystem`](crate::ReactionSystem).
#[derive(Debug, Error)]
pub enum ReactionNetworkError {
    /// A species name was referenced (e.g. in the textual DSL or an initial
    /// state) that was never declared.
    #[error("unknown species referenced by name: {0}")]
    UnknownSpecies(String),

    /// A parameter name was referenced that was never declared.
    #[error("unknown parameter referenced by name: {0}")]
    UnknownParameter(String),

    /// A reaction was added with neither reactants nor products.
    #[error("reaction has no reactants and no products (empty reaction)")]
    EmptyReaction,

    /// A `MassAction` rate law referenced a rate-constant parameter that was
    /// never declared.
    #[error("rate-constant parameter {0} was never defined")]
    UndefinedRateConstant(String),

    /// The initial state passed to [`crate::ReactionSystem::to_ode_problem`] does not
    /// have one entry per species.
    #[error("initial state length {got} does not match number of species {expected}")]
    StateDimension { got: usize, expected: usize },

    /// The initial state for a stochastic (Gillespie) simulation was invalid:
    /// species counts must be finite and non-negative.
    #[error("invalid stochastic initial state: {0}")]
    InvalidState(String),

    /// The textual DSL could not be parsed.
    #[error("invalid reaction-network DSL: {0}")]
    Dsl(String),

    /// The SBML document could not be parsed by the minimal reader in
    /// [`crate::sbml`] (malformed XML, or outside the supported subset).
    #[error("invalid or unsupported SBML: {0}")]
    Sbml(String),

    /// Wraps an error from the underlying ODE solver ([`tpt_sci_ode`]).
    #[error("underlying ODE error: {0}")]
    Ode(#[from] tpt_sci_ode::OdeError),
}
