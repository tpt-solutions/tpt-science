//! Minimal SBML (Systems Biology Markup Language) reader.
//!
//! This is a hand-rolled scanner over well-formed XML — not a general,
//! validating XML/MathML parser — sized for the narrow, practical subset of
//! SBML Level 2/3 core that typical toy mass-action models use. No new
//! dependency was added for this: the tag/attribute scanner below is
//! sufficient for the supported subset and keeps the crate's dependency
//! graph unchanged.
//!
//! # Supported
//!
//! * `<model><listOfSpecies><species id=".." initialAmount=".."/>` (or
//!   `initialConcentration`, treated identically — this reader does not
//!   model compartment volumes, so "amount" and "concentration" are not
//!   distinguished).
//! * `<listOfParameters><parameter id=".." value=".."/>`, both at the
//!   `<model>` level (global parameters) and nested inside a `<kineticLaw>`
//!   (SBML Level 2 `<listOfParameters>` / Level 3
//!   `<listOfLocalParameters><localParameter .../>`). Local parameters are
//!   hoisted into the network's single global parameter table by their
//!   `id`; a local parameter that collides with another reaction's local
//!   parameter of the same `id` is **not** supported (last write wins).
//! * `<listOfReactions><reaction><listOfReactants>` /
//!   `<listOfProducts><speciesReference species=".." stoichiometry=".."/>`
//!   (`stoichiometry` defaults to `1` when omitted). `<listOfModifiers>`
//!   references are ignored (modifiers neither consume nor produce).
//! * `<kineticLaw><math>...</math></kineticLaw>`, restricted to a **product
//!   (mass-action) form**: a rate constant — either a `<ci>` naming a
//!   declared parameter, or a bare `<cn>` numeric literal — multiplied
//!   (`<times/>` must appear somewhere in the expression whenever there is
//!   more than one factor) by the reactant concentrations. The reactant
//!   factors themselves are **not** interpreted from the MathML: the
//!   reaction's declared `speciesReference` stoichiometry (from
//!   `listOfReactants`) is what drives the mass-action exponent, exactly as
//!   with the programmatic [`crate::ReactionNetwork`] API. This means a
//!   `<kineticLaw>` that multiplies a reactant by an *unusual* power (one
//!   that does not match its `stoichiometry` attribute) will silently use
//!   the declared stoichiometry instead of the MathML exponent.
//!
//! # Not supported
//!
//! * Compartments / volumes, units, `boundaryCondition`, `constant` flags —
//!   parsed structurally (skipped) but never used.
//! * Reversible reactions: `reversible="true"` is accepted but the reverse
//!   direction is **not** auto-generated — write it as a second
//!   `<reaction>` with its own `<kineticLaw>`.
//! * General MathML: no `<plus/>`, `<divide/>`, `<power/>`, Michaelis–Menten
//!   or Hill forms, `<piecewise/>`, function definitions, etc. A
//!   `<kineticLaw>` containing more than one `<ci>`/`<cn>` factor without a
//!   `<times/>` is rejected with [`crate::ReactionNetworkError::Sbml`]
//!   rather than silently misinterpreted.
//! * `<listOfRules>`, `<listOfEvents>`, `<listOfConstraints>`,
//!   `<listOfFunctionDefinitions>`, `<listOfUnitDefinitions>`,
//!   `<listOfCompartments>` are ignored entirely.
//! * XML namespaces are ignored (tags are matched by local name only, e.g.
//!   `sbml:species` and `species` are treated the same), so Level 2 and
//!   Level 3 documents parse the same way; XML entities other than the
//!   ones needed to read plain numeric/identifier text are not decoded.

use std::collections::{HashMap, HashSet};

use crate::error::ReactionNetworkError;
use crate::model::{RateLaw, ReactionNetwork};

/// Result of parsing an SBML document: the compiled [`ReactionNetwork`] plus
/// the per-species initial amounts declared in `<listOfSpecies>`, ready to
/// be passed to [`crate::ReactionSystem::initial_state`] after
/// [`ReactionNetwork::build`].
#[derive(Debug, Clone)]
pub struct SbmlModel {
    /// The parsed network (species, parameters, reactions).
    pub network: ReactionNetwork,
    /// `(species_name, initial_amount)` pairs, in document order.
    pub initial_amounts: Vec<(String, f64)>,
}

impl ReactionNetwork {
    /// Parse an SBML XML document (the subset documented in
    /// [`crate::sbml`]) into a new network.
    ///
    /// # Errors
    ///
    /// Returns [`ReactionNetworkError::Sbml`] if the document is not
    /// well-formed XML, or uses SBML features outside the supported subset
    /// (e.g. a `<kineticLaw>` that is not a simple mass-action product, or a
    /// `<reaction>` with no `<kineticLaw>` at all).
    pub fn from_sbml(xml: &str) -> Result<SbmlModel, ReactionNetworkError> {
        parse_sbml(xml)
    }
}

/// One raw XML event produced by [`xml_events`].
#[derive(Debug)]
enum XmlEvent<'a> {
    Start {
        name: &'a str,
        attrs: HashMap<&'a str, String>,
        self_closing: bool,
    },
    End {
        name: &'a str,
    },
    Text(&'a str),
}

/// A reaction accumulated while scanning between `<reaction>` and
/// `</reaction>`.
#[derive(Default)]
struct PendingReaction {
    reactants: Vec<(usize, f64)>,
    products: Vec<(usize, f64)>,
    ci_terms: Vec<String>,
    cn_terms: Vec<f64>,
    has_times: bool,
}

fn parse_sbml(xml: &str) -> Result<SbmlModel, ReactionNetworkError> {
    let events = xml_events(xml)?;
    let mut net = ReactionNetwork::new();
    let mut known_params: HashSet<String> = HashSet::new();
    let mut initial_amounts: Vec<(String, f64)> = Vec::new();

    let mut in_reactants = false;
    let mut in_products = false;
    let mut in_kinetic_law = false;
    let mut in_ci = false;
    let mut in_cn = false;
    let mut current: Option<PendingReaction> = None;
    let mut anon_param_counter = 0usize;

    for ev in events {
        match ev {
            XmlEvent::Start {
                name,
                attrs,
                self_closing,
            } => {
                let local = local_name(name);
                match local {
                    "species" => {
                        let id = attrs.get("id").ok_or_else(|| {
                            ReactionNetworkError::Sbml("<species> is missing an id".to_string())
                        })?;
                        net.species(id);
                        let amount = attr_f64(&attrs, "initialAmount")?
                            .or(attr_f64(&attrs, "initialConcentration")?)
                            .unwrap_or(0.0);
                        initial_amounts.push((id.clone(), amount));
                    }
                    "parameter" | "localParameter" => {
                        let id = attrs.get("id").ok_or_else(|| {
                            ReactionNetworkError::Sbml("<parameter> is missing an id".to_string())
                        })?;
                        let value = attr_f64(&attrs, "value")?.unwrap_or(0.0);
                        net.parameter(id, value);
                        known_params.insert(id.clone());
                    }
                    "reaction" => {
                        current = Some(PendingReaction::default());
                    }
                    "listOfReactants" => in_reactants = true,
                    "listOfProducts" => in_products = true,
                    "kineticLaw" => in_kinetic_law = true,
                    "times" if in_kinetic_law => {
                        if let Some(r) = current.as_mut() {
                            r.has_times = true;
                        }
                    }
                    "speciesReference" if in_reactants || in_products => {
                        let sp = attrs.get("species").ok_or_else(|| {
                            ReactionNetworkError::Sbml(
                                "<speciesReference> is missing a species attribute".to_string(),
                            )
                        })?;
                        let idx = net.species(sp);
                        let stoich = attr_f64(&attrs, "stoichiometry")?.unwrap_or(1.0);
                        if let Some(r) = current.as_mut() {
                            if in_reactants {
                                r.reactants.push((idx, stoich));
                            } else {
                                r.products.push((idx, stoich));
                            }
                        }
                    }
                    "ci" if in_kinetic_law => in_ci = true,
                    "cn" if in_kinetic_law => in_cn = true,
                    _ => {}
                }
                // Self-closing tags never receive a matching End event, so
                // any state they set must be undone immediately.
                if self_closing {
                    match local {
                        "listOfReactants" => in_reactants = false,
                        "listOfProducts" => in_products = false,
                        "kineticLaw" => in_kinetic_law = false,
                        "ci" => in_ci = false,
                        "cn" => in_cn = false,
                        _ => {}
                    }
                }
            }
            XmlEvent::End { name } => {
                let local = local_name(name);
                match local {
                    "listOfReactants" => in_reactants = false,
                    "listOfProducts" => in_products = false,
                    "kineticLaw" => in_kinetic_law = false,
                    "ci" => in_ci = false,
                    "cn" => in_cn = false,
                    "reaction" => {
                        let r = current.take().ok_or_else(|| {
                            ReactionNetworkError::Sbml(
                                "</reaction> without a matching <reaction>".to_string(),
                            )
                        })?;
                        let rate =
                            resolve_rate_law(&r, &known_params, &mut net, &mut anon_param_counter)?;
                        net.reaction(&r.reactants, &r.products, rate);
                    }
                    _ => {}
                }
            }
            XmlEvent::Text(text) => {
                if in_ci {
                    if let Some(r) = current.as_mut() {
                        r.ci_terms.push(text.trim().to_string());
                    }
                } else if in_cn {
                    if let Some(r) = current.as_mut() {
                        let v: f64 = text.trim().parse().map_err(|_| {
                            ReactionNetworkError::Sbml(format!("invalid <cn> literal: {text}"))
                        })?;
                        r.cn_terms.push(v);
                    }
                }
            }
        }
    }

    Ok(SbmlModel {
        network: net,
        initial_amounts,
    })
}

/// Decide the [`RateLaw`] for a reaction from the `<ci>`/`<cn>` factors and
/// `<times/>` marker collected from its `<kineticLaw>`. See the module docs
/// for exactly what shape of `<kineticLaw>` this recognises.
fn resolve_rate_law(
    r: &PendingReaction,
    known_params: &HashSet<String>,
    net: &mut ReactionNetwork,
    anon_param_counter: &mut usize,
) -> Result<RateLaw, ReactionNetworkError> {
    let total_factors = r.ci_terms.len() + r.cn_terms.len();
    if total_factors == 0 {
        return Err(ReactionNetworkError::Sbml(
            "<kineticLaw> has no <ci> or <cn> terms; a mass-action rate constant is required"
                .to_string(),
        ));
    }
    if total_factors > 1 && !r.has_times {
        return Err(ReactionNetworkError::Sbml(
            "<kineticLaw> combines more than one term without a <times/>; only simple \
             mass-action product expressions are supported"
                .to_string(),
        ));
    }

    let param_matches: Vec<&String> = r
        .ci_terms
        .iter()
        .filter(|c| known_params.contains(*c))
        .collect();
    match param_matches.len() {
        1 => Ok(RateLaw::mass_action(param_matches[0])),
        0 => {
            if let Some(&literal) = r.cn_terms.first() {
                *anon_param_counter += 1;
                let name = format!("_sbml_k{anon_param_counter}");
                net.parameter(&name, literal);
                Ok(RateLaw::mass_action(&name))
            } else {
                Err(ReactionNetworkError::Sbml(format!(
                    "could not find a rate-constant parameter among <ci> terms {:?} in \
                     <kineticLaw>; expected exactly one to name a declared parameter",
                    r.ci_terms
                )))
            }
        }
        _ => Err(ReactionNetworkError::Sbml(format!(
            "<kineticLaw> has ambiguous rate constant: multiple <ci> terms {:?} name declared \
             parameters",
            param_matches
        ))),
    }
}

/// Strip an XML namespace prefix (`sbml:species` -> `species`); namespaces
/// are otherwise ignored by this reader.
fn local_name(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

/// Read a named attribute as `f64`, if present.
fn attr_f64(attrs: &HashMap<&str, String>, key: &str) -> Result<Option<f64>, ReactionNetworkError> {
    match attrs.get(key) {
        None => Ok(None),
        Some(v) => v.trim().parse::<f64>().map(Some).map_err(|_| {
            ReactionNetworkError::Sbml(format!("invalid numeric attribute {key}={v:?}"))
        }),
    }
}

/// Tokenize `xml` into a flat sequence of start/end/text events. Comments
/// (`<!-- ... -->`), processing instructions (`<?xml ... ?>`), and
/// `<!DOCTYPE ...>` declarations are skipped. This is a scanner, not a
/// validating parser: it assumes well-formed input and does not attempt to
/// recover from malformed markup beyond returning
/// [`ReactionNetworkError::Sbml`].
fn xml_events(xml: &str) -> Result<Vec<XmlEvent<'_>>, ReactionNetworkError> {
    let mut events = Vec::new();
    let bytes = xml.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        if bytes[i] == b'<' {
            if xml[i..].starts_with("<!--") {
                let rel = xml[i..].find("-->").ok_or_else(|| {
                    ReactionNetworkError::Sbml("unterminated comment".to_string())
                })?;
                i += rel + 3;
                continue;
            }
            if xml[i..].starts_with("<?") {
                let rel = xml[i..].find("?>").ok_or_else(|| {
                    ReactionNetworkError::Sbml("unterminated processing instruction".to_string())
                })?;
                i += rel + 2;
                continue;
            }
            if xml[i..].starts_with("<!") {
                let rel = xml[i..].find('>').ok_or_else(|| {
                    ReactionNetworkError::Sbml("unterminated <! declaration".to_string())
                })?;
                i += rel + 1;
                continue;
            }
            // Find the tag's closing '>', respecting quoted attribute values
            // (which may themselves contain '>').
            let mut j = i + 1;
            let mut in_quote: Option<u8> = None;
            while j < n {
                let c = bytes[j];
                match in_quote {
                    Some(q) if c == q => in_quote = None,
                    Some(_) => {}
                    None if c == b'"' || c == b'\'' => in_quote = Some(c),
                    None if c == b'>' => break,
                    None => {}
                }
                j += 1;
            }
            if j >= n {
                return Err(ReactionNetworkError::Sbml("unterminated tag".to_string()));
            }
            let tag_str = &xml[i + 1..j];
            i = j + 1;
            if let Some(rest) = tag_str.strip_prefix('/') {
                events.push(XmlEvent::End { name: rest.trim() });
                continue;
            }
            let trimmed = tag_str.trim_end();
            let self_closing = trimmed.ends_with('/');
            let core = if self_closing {
                trimmed[..trimmed.len() - 1].trim_end()
            } else {
                tag_str
            };
            let (name, attrs) = parse_tag_core(core)?;
            events.push(XmlEvent::Start {
                name,
                attrs,
                self_closing,
            });
        } else {
            let end = xml[i..].find('<').map_or(n, |k| i + k);
            let text = &xml[i..end];
            if !text.trim().is_empty() {
                events.push(XmlEvent::Text(text));
            }
            i = end;
        }
    }
    Ok(events)
}

/// Parse a tag's interior (name plus `key="value"` attributes, already
/// stripped of the leading `<`/`</`, trailing `>`/`/`, and any namespace
/// declarations are kept as ordinary attributes and simply ignored by
/// callers).
fn parse_tag_core(core: &str) -> Result<(&str, HashMap<&str, String>), ReactionNetworkError> {
    let core = core.trim();
    let name_end = core.find(|c: char| c.is_whitespace()).unwrap_or(core.len());
    let name = &core[..name_end];
    let mut attrs = HashMap::new();
    let mut rest = core[name_end..].trim_start();
    while !rest.is_empty() {
        let eq = rest.find('=').ok_or_else(|| {
            ReactionNetworkError::Sbml(format!("malformed attribute near '{rest}' in <{core}>"))
        })?;
        let key = rest[..eq].trim();
        let after_eq = rest[eq + 1..].trim_start();
        let quote = after_eq.chars().next().ok_or_else(|| {
            ReactionNetworkError::Sbml(format!("missing attribute value in <{core}>"))
        })?;
        if quote != '"' && quote != '\'' {
            return Err(ReactionNetworkError::Sbml(format!(
                "expected a quoted attribute value in <{core}>"
            )));
        }
        let value_body = &after_eq[1..];
        let end_rel = value_body.find(quote).ok_or_else(|| {
            ReactionNetworkError::Sbml(format!("unterminated attribute value in <{core}>"))
        })?;
        attrs.insert(key, value_body[..end_rel].to_string());
        rest = value_body[end_rel + 1..].trim_start();
    }
    Ok((name, attrs))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BIRTH_DEATH_SBML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="birth_death">
    <listOfSpecies>
      <species id="A" initialAmount="5"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="2"/>
      <parameter id="k2" value="1"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="birth" reversible="false">
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <ci> k1 </ci>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="death" reversible="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1"/>
        </listOfReactants>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <times/>
              <ci> k2 </ci>
              <ci> A </ci>
            </apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"#;

    const A_PLUS_B_TO_C_SBML: &str = r#"<sbml level="2" version="4">
  <model>
    <listOfSpecies>
      <species id="A" initialAmount="10"/>
      <species id="B" initialAmount="20"/>
      <species id="C" initialAmount="0"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="combine">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1"/>
          <speciesReference species="B" stoichiometry="1"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="C" stoichiometry="1"/>
        </listOfProducts>
        <kineticLaw>
          <math>
            <apply>
              <times/>
              <cn> 0.5 </cn>
              <ci> A </ci>
              <ci> B </ci>
            </apply>
          </math>
          <listOfParameters>
            <parameter id="unused" value="99"/>
          </listOfParameters>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"#;

    #[test]
    fn birth_death_round_trip() {
        let parsed = ReactionNetwork::from_sbml(BIRTH_DEATH_SBML).unwrap();
        let sys = parsed.network.build().unwrap();
        assert_eq!(sys.species_names(), &["A".to_string()]);
        assert_eq!(sys.n_reactions(), 2);
        assert_eq!(sys.parameter("k1"), Some(2.0));
        assert_eq!(sys.parameter("k2"), Some(1.0));
        assert_eq!(parsed.initial_amounts, vec![("A".to_string(), 5.0)]);

        // Stoichiometry: birth reaction produces A (+1), death consumes A (-1).
        let s = sys.stoichiometry_matrix();
        assert_eq!(s, vec![vec![1.0, -1.0]]);

        // Rates: birth = k1 = 2, death = k2 * A.
        let rates = sys.reaction_rates(&[3.0]);
        assert_eq!(rates[0], 2.0);
        assert_eq!(rates[1], 1.0 * 3.0);
    }

    #[test]
    fn a_plus_b_to_c_round_trip() {
        let parsed = ReactionNetwork::from_sbml(A_PLUS_B_TO_C_SBML).unwrap();
        let sys = parsed.network.build().unwrap();
        assert_eq!(sys.n_species(), 3);
        assert_eq!(sys.n_reactions(), 1);
        let a = sys.species_index("A").unwrap();
        let b = sys.species_index("B").unwrap();
        let c = sys.species_index("C").unwrap();

        let s = sys.stoichiometry_matrix();
        assert_eq!(s[a][0], -1.0);
        assert_eq!(s[b][0], -1.0);
        assert_eq!(s[c][0], 1.0);

        // The literal <cn>0.5</cn> became an anonymous rate constant.
        let rates = sys.reaction_rates(&[2.0, 4.0, 0.0]);
        assert_eq!(rates[0], 0.5 * 2.0 * 4.0);

        assert_eq!(
            parsed
                .initial_amounts
                .iter()
                .map(|(n, v)| (n.as_str(), *v))
                .collect::<Vec<_>>(),
            vec![("A", 10.0), ("B", 20.0), ("C", 0.0)]
        );
    }

    #[test]
    fn missing_kinetic_law_is_rejected() {
        let xml = r#"<sbml><model>
            <listOfSpecies><species id="A" initialAmount="1"/></listOfSpecies>
            <listOfReactions>
              <reaction id="r">
                <listOfReactants><speciesReference species="A"/></listOfReactants>
              </reaction>
            </listOfReactions>
        </model></sbml>"#;
        assert!(matches!(
            ReactionNetwork::from_sbml(xml),
            Err(ReactionNetworkError::Sbml(_))
        ));
    }

    #[test]
    fn non_product_kinetic_law_is_rejected() {
        // Two <ci> terms with no <times/> — not a supported mass-action shape.
        let xml = r#"<sbml><model>
            <listOfSpecies>
              <species id="A" initialAmount="1"/>
              <species id="B" initialAmount="1"/>
            </listOfSpecies>
            <listOfParameters><parameter id="k" value="1"/></listOfParameters>
            <listOfReactions>
              <reaction id="r">
                <listOfReactants><speciesReference species="A"/></listOfReactants>
                <listOfProducts><speciesReference species="B"/></listOfProducts>
                <kineticLaw><math>
                  <apply><plus/><ci>k</ci><ci>A</ci></apply>
                </math></kineticLaw>
              </reaction>
            </listOfReactions>
        </model></sbml>"#;
        assert!(matches!(
            ReactionNetwork::from_sbml(xml),
            Err(ReactionNetworkError::Sbml(_))
        ));
    }

    #[test]
    fn malformed_xml_is_rejected() {
        let xml = "<sbml><model><listOfSpecies><species id=\"A\"</model></sbml>";
        assert!(ReactionNetwork::from_sbml(xml).is_err());
    }
}
