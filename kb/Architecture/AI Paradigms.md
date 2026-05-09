---
tags: [architecture, fuzzy, ai-history, concepts]
source: llm
date: 2026-05-09
---
# AI Paradigms

Conceptual map of major AI paradigms — how they differ, where they overlap, and
where fuzzy logic fits. This background informs Prism's design decisions around
inference and interpretability.

## Summary

| Paradigm | Core Idea |
|---|---|
| Symbolic AI | Intelligence via explicit rules and logic |
| Expert Systems | Symbolic AI specialized for domain expertise |
| Decision Trees | Learned symbolic rules from data |
| Fuzzy Logic | Reasoning with partial truth instead of binary truth |
| Modern ML | Learn statistical patterns from data |
| Generative AI | Learn distributions that can create new content |

They overlap more than people often realize.

---

## 1. Symbolic AI

Also called: GOFAI ("Good Old-Fashioned AI"), rule-based AI, knowledge-based AI.

**Core philosophy:** represent intelligence explicitly using symbols, logic,
rules, and facts.

```text
IF fever AND cough THEN flu
```

Typical system components:

- Knowledge base
- Rule engine
- Inference engine

Symbolic AI remains useful where explainability, deterministic behavior, safety,
or limited data are constraints (medical rules, finance compliance, industrial
automation, configuration systems).

---

## 2. Expert Systems

Expert systems are a major application of Symbolic AI. Classic examples:
medical diagnosis, industrial control, troubleshooting. Famous historical
systems: MYCIN, XCON.

Classic expert systems use binary logic (`symptom = true/false`). Fuzzy expert
systems extend this to graded certainty (`symptom = 0.73 true`), making fuzzy
logic a softer, uncertainty-aware evolution of symbolic expert systems.

---

## 3. Why Fuzzy Logic Emerged

Real-world reasoning is rarely binary. Humans say "somewhat hot", "very risky",
"slightly damaged". Traditional symbolic AI struggles with this ambiguity.

**Classical logic:**

```text
temperature > 30 => HOT = true   # hard threshold
```

**Fuzzy logic:**

```text
temperature can be 40% warm AND 70% hot simultaneously
```

Fuzzy systems are excellent for control, heuristics, and human-like reasoning
because they avoid sharp discontinuities.

---

## 4. Decision Trees

Decision trees sit between symbolic AI and machine learning. They produce
symbolic-looking logic:

```text
IF income > 50k
  IF age < 30
    approve
```

Unlike expert systems, the rules are *statistically learned* from data, not
handcrafted. Trees are interpretable, symbolic, and explainable — but
data-driven. "Learned symbolic AI" is a useful modern framing.

---

## 5. Fuzzy Logic vs Decision Trees

| | Decision Tree | Fuzzy System |
|---|---|---|
| Boundaries | Hard (`age < 30`) | Soft (`young = 0.6, middle_aged = 0.4`) |
| Danger check | `IF speed > 100 THEN danger` | danger increases gradually with speed |

Fuzzy systems avoid the sharp discontinuities that decision trees inherit from
binary splits.

---

## 6. Relationship to Modern Machine Learning

Modern ML abandoned explicit symbolic rules in favor of statistical
relationships, latent representations, and optimization surfaces. Knowledge is
encoded implicitly in parameters (weights, embeddings, vector spaces) rather
than as explicit `IF X THEN Y` rules.

**Why ML won for perception/language/vision:** better scaling to noisy data,
massive datasets, and ambiguous feature spaces.

**Why symbolic AI never died:** determinism, explainability, safety, and
low-data regimes still favor it. Medical rules, compliance, industrial
automation.

---

## 7. Where Fuzzy Logic Fits Today

Fuzzy logic is a hybrid:

- Symbolic and interpretable (rules are human-readable)
- Uncertainty-aware (continuous truth values)
- Not statistical learning by default

**Strengths:**

- Human-readable rules
- Robust control (appliances, vehicles, robotics)
- Small-data friendly — works with expert knowledge and heuristics

**Weaknesses:**

- Hard to scale manually
- Hard to optimize globally
- Does not learn automatically (see ANFIS below)

Fuzzy systems remain valuable when you need interpretability, continuous
heuristics, low compute, deterministic control, or expert-guided behavior.
Especially in embedded systems, industrial control, robotics, edge AI, game AI.

---

## 8. ANFIS — Fuzzy + Neural Networks

ANFIS combines fuzzy rules with neural learning: humans define structure, ML
tunes parameters. It bridges symbolic and statistical paradigms.

ANFIS is deliberately out of scope for Prism v1. See
[[Fuzzy Logic Capability#Slice Mapping]] for the promotion rule.

---

## 9. Non-Generative (Discriminative) AI

"Non-generative AI" usually refers to predictive/discriminative models:
classification, forecasting, ranking, anomaly detection, recommendation.

| Model | Nature |
|---|---|
| Decision Trees | Interpretable ML |
| Random Forests | Ensemble trees |
| Gradient Boosting (XGBoost, LightGBM, CatBoost) | High-performance structured ML |
| SVMs | Geometric classification |
| Neural Networks | Statistical function approximation |
| Fuzzy Systems | Rule-based uncertainty reasoning |

For tabular/business data (finance, fraud, insurance, analytics), gradient
boosted trees often outperform deep learning.

---

## 10. Neuro-Symbolic AI

A major modern trend: combine symbolic reasoning/logic/explainability with
statistical learning/pattern recognition/scalability.

Examples:

- **LLM + tool use** — language model interprets language; symbolic engine
  executes logic
- **Neural + knowledge graph** — neural network for perception; graph/rules for
  reasoning
- **Fuzzy + ML** — ML tunes fuzzy system parameters (ANFIS)

---

## 11. Big Picture Evolution

```text
Symbolic AI
    ↓
Expert Systems
    ↓
Fuzzy Logic
    ↓
Statistical ML
    ↓
Deep Learning
    ↓
Generative AI
    ↓
Neuro-symbolic hybrids
```

These are not replacements — modern AI stacks often combine multiple paradigms.

---

## 12. Mental Model

| System Type | Best At |
|---|---|
| Symbolic AI | Explicit reasoning |
| Expert Systems | Codified expertise |
| Fuzzy Logic | Vague human reasoning |
| Decision Trees | Interpretable learned rules |
| Neural Networks | Perception / patterns |
| Generative AI | Synthesis / creation |

**One important insight:** Modern AI shifted from "How do we encode
intelligence?" to "How do we learn intelligence from data?" Fuzzy logic sits in
the middle — structured like symbolic AI, smooth like statistical systems. That
is why it remains conceptually important even as statistical ML dominates.

See also: [[Fuzzy Logic Capability]], [[Surface]]
