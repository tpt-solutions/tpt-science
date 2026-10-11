# tpt-sci-units

Typed SI quantities and the physical constants shared across tpt-science:
the molar gas constant, the Stefan–Boltzmann constant and Earth's gravitational
parameter, each defined once with its unit in the type.

The quantity types are newtypes over `f64`. Use `.value()` to get the raw number.
The kinetics, climate and astro crates take their constants from here.
