//! SIR epidemic model built with the reaction-network DSL and integrated via
//! `tpt-sci-ode`, reporting the peak infected fraction.
use tpt_sci_ode::Method;
use tpt_sci_reaction_network::ReactionNetwork;

fn main() {
    let mut model = ReactionNetwork::from_dsl(
        "beta, S + I --> 2 I
         gamma, I --> R",
    )
    .unwrap();
    model.set_parameter("beta", 0.002).unwrap();
    model.set_parameter("gamma", 0.4).unwrap();

    let y0 = model
        .initial_state(&[("S", 990.0), ("I", 10.0), ("R", 0.0)])
        .unwrap();
    let prob = model.to_ode_problem(&y0, 0.0).unwrap();
    let i = model.species_index("I").unwrap();

    println!("  t =  0.0  I = {:.2}", y0[i]);

    // Sample the curve and track the peak infected count. (diffsol requires a
    // strictly positive integration span, so we start sampling at t = 1.0.)
    let times = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 25.0, 40.0, 60.0];
    let mut peak = y0[i];
    for &t in &times {
        let y = prob.solve(Method::Bdf, t).unwrap();
        peak = peak.max(y[i]);
        println!("  t = {t:5.1}  I = {:.2}", y[i]);
    }
    println!("Peak infected = {peak:.2} (of 1000)");
}
