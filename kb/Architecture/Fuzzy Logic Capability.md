---
tags: [architecture, fuzzy, math]
source: mixed
date: 2026-05-08
updated: 2026-05-09
---
# Fuzzy Logic Capability

Fuzzy logic is a Prism capability for representing graded truth, perception,
and expectation states. It is exposed as reusable `prism::fuzzy` code and as a
Converge pack adapter through `FuzzyInferencePack`.

## Mathematical Model

Classical logic maps a proposition to either false or true:

```text
false = 0
true  = 1
```

Fuzzy logic allows degrees of truth between 0 and 1:

```text
0.0 = completely false
0.2 = weakly true
0.7 = mostly true
1.0 = completely true
```

The core object is a membership function:

```text
mu_A(x): X -> [0, 1]
```

This describes how strongly value `x` belongs to fuzzy set `A`. For example,
`mu_warm(55 C) = 0.7` says that 55 C belongs to the set `warm` with degree
0.7.

## Building Blocks

Membership functions map crisp values to membership degrees. Prism supports
triangular, trapezoidal, left-shoulder, right-shoulder, and Gaussian
functions. The Gaussian shape is `μ(x) = exp(-((x − c)² / (2σ²)))` with
`σ > 0`.

Fuzzy sets replace binary membership:

```text
x in A
```

with graded membership:

```text
mu_A(x) = 0.73
```

Fuzzy operators combine memberships:

```text
AND = min(a, b)
OR  = max(a, b)
NOT = 1 - a
```

Other t-norms, s-norms, and probabilistic operators can be added later without
changing the Converge pack boundary.

Fuzzy inference evaluates rules:

```text
IF service is HIGH AND wait_time is LOW
THEN satisfaction is VERY_HIGH
```

Each rule fires with a strength in `[0, 1]`. Consequents are aggregated by max
membership in the first implementation.

Defuzzification converts fuzzy outputs into a crisp value, such as `82/100`.
Mamdani inference returns explainable memberships and activated-rule traces
by default; defuzzification is offered as a separate, opt-in pass via
`prism::fuzzy::defuzzify_mamdani(output, variables, output_variable, domain,
method)` so the activated-rule trace is preserved on the original output.

Available defuzzification methods:

- **Centroid** — center of mass of the aggregated output set; the textbook
  default for Mamdani.
- **Bisector** — point that divides the area in two equal parts.
- **Mean of maxima (MoM)** — average of x values where the aggregated set
  reaches its maximum membership.
- **Height** — the single x with maximum aggregated membership.
- **Weighted average** — used by Sugeno inference internally; exposed as
  `weighted_average(rules)` for callers that have crisp `(strength, value)`
  pairs.

## Slice Mapping

Within the broader fuzzy-logic landscape, Prism v1 ships two of the five
classical inference systems plus a defuzzification module. Tsukamoto, Type-2,
and ANFIS remain non-goals for v1.

### Reference matrix — fuzzy inference systems

| System | Output type | Defuzzification | Complexity | Main strength | In Prism v1 |
|---|---|---|---|---|---|
| **Mamdani** | Fuzzy set | Yes (opt-in) | Medium | Human-readable | ✓ shipped |
| **Sugeno** | Equation / constant | Weighted average | Low–Medium | Fast & ML-friendly | ✓ shipped |
| **Tsukamoto** | Crisp from monotonic fuzzy set | Weighted average | Medium | Smooth outputs | — out of scope |
| **Type-2** | Fuzzy uncertainty sets | Type reduction | High | Handles uncertainty | — out of scope |
| **ANFIS** | Learned Sugeno | Weighted average | Very high | Learns from data | — out of scope |

### In scope (v1)

- **Mamdani FIS.** Rule consequents are linguistic terms over fuzzy sets.
  Output is a `FuzzyInferenceOutput` with input memberships, output
  memberships, activated rules with strengths, confidence, and total rule
  count. `FuzzyInferencePack` exposes it as a Converge pack.
- **Sugeno (Takagi–Sugeno) FIS.** Rule consequents are mathematical
  functions of inputs — order-0 (constant) or order-1 (linear:
  `intercept + Σ coefficient_i × input_i`). Inference returns a single
  weighted-average crisp output along with per-rule firing strengths and
  consequent values. `SugenoInferencePack` exposes it as a Converge pack.
- **Membership functions.** Triangular, trapezoidal, left-shoulder,
  right-shoulder, Gaussian.
- **Operators.** `is`, `and` (min), `or` (max), `not` (1 − a).
- **Defuzzification methods.** Centroid, Bisector, MeanOfMaxima, Height,
  WeightedAverage. Available as a separate pass for Mamdani; baked into
  Sugeno inference.
- Per-rule weights and per-firing strength on activated rules are reported
  by both engines.

### Deliberately out of scope (non-goals for v1)

- **Tsukamoto FIS.** Smooth outputs via weighted average over crisp values
  derived from monotonic consequent MFs. Requires monotonic-MF validation
  and a different aggregation rule. Promote when an app needs the smooth
  output property and Sugeno's linear consequents are not sufficient.
- **Type-2 fuzzy logic.** Interval or general Type-2 sets carry membership
  uncertainty and require a type-reduction step (e.g. Karnik–Mendel). High
  computational cost. Promote when an app must reason about *uncertainty
  in the rule definitions themselves*, not just in the inputs.
- **ANFIS / neuro-fuzzy / learned fuzzy.** Trained Sugeno via gradient
  descent on Gaussian-MF parameters. Requires a training pipeline (Burn
  integration, training-agent infrastructure). Promote when an app has the
  data and labels to learn rules from, rather than authoring them.

### The two niches the v1 slice serves

- **Mamdani niche** — consultative / interpretive workflows where the
  rulebook itself is part of the artifact a domain expert must defend.
  Preserving "rule R3 fired at 0.84" is more valuable than collapsing the
  result to one number. Small-N qualitative inputs become graded,
  inspectable decision states.
- **Sugeno niche** — fast, numerically smooth control / scoring
  workflows where the consequent is a known function of inputs (often
  linear in the ML-friendly case) and a single crisp output is the
  product. Lower complexity than Mamdani, easier to compose with
  ranking / regression downstream.

### Promotion rule

Adding a new slice (Tsukamoto, Type-2, ANFIS) requires a concrete pull from
an app or engagement that demonstrably needs that variant's distinguishing
property — not analogy to fuzzy-logic toolkits or completeness for its own
sake.

## Boundary

`prism::fuzzy` owns reusable fuzzy math, typed rule evaluation, and inference
outputs.

`FuzzyInferencePack` owns the Converge pack wrapper so fuzzy inference can
participate in formations as a suggestor.

Products own domain-specific variables, membership functions, and rules.

`arbiter` still owns hard policy, `ferrox` still owns hard constraints and
optimization, and `mnemos` still owns memory and recall.

See also: [[Surface]]
