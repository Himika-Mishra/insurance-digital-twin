# Insurance Portfolio Digital Twin

Synthetic personal-lines insurance portfolio built as a **governed digital twin**,  
with dataset freezing, validation gates, and actuarial realism.

This repository evolves in **phases**, each adding analytical depth while preserving
governance, reproducibility, and auditability.

Designed to mirror how regulated insurance analytics platforms are built internally,
rather than how public modelling demos are typically presented.

---

## Project Phases

This project is structured as a **multi-phase insurance analytics build**, where each
phase produces a stable, defensible artefact before moving forward.

---

## Phase 1 — Synthetic Insurance Universe & Governance (v0.1)

The focus of Phase 1 is **not modelling** —  
it is **data generation, governance, validation, and auditability**.

Before pricing, fraud, forecasting, or scenario analysis can be trusted,  
the underlying dataset must be **frozen, reproducible, and defensible**.

That is what Phase 1 delivers.

### Why this project exists

In real insurance environments, analytical credibility depends on:
- reproducibility
- traceability
- controlled imperfections
- governance before modelling

Most public analytics projects skip these steps.

This project does not.

### Phase 1 scope

**Delivered in this repository:**

✔ Synthetic personal-lines insurance universe  
✔ Policyholders, policies, claims, macro environment  
✔ Explicit modelling assumptions (documented in `config.py`)  
✔ Controlled anomaly injection (real-world messiness)  
✔ Validation gates (actuarial sanity checks)  
✔ Dataset freeze with manifest and cryptographic hashes  
✔ Auditable, versioned data artefact  

**Explicitly not included yet:**
- pricing models
- fraud models
- scenario simulators
- dashboards or UI

These are added incrementally in later phases.

### What makes this different

This repository treats synthetic data as a **governed asset**, not a toy dataset.

It includes:
- deterministic generation via fixed random seeds
- hash-based dataset locking
- validation checks aligned to actuarial practice
- anomaly rates that are intentional, rare, and bounded

This mirrors how internal insurance analytics platforms are built.

---

## Phase 2 — Portfolio Mix & Premium Distributions (Pricing Context) (v0.2)

Phase 2 builds **pricing context** on top of the frozen dataset produced in Phase 1.

No data is regenerated or modified in this phase.

The objective is to answer the questions that pricing and actuarial teams ask
*before* loss ratio modelling or rate changes:

- What is the portfolio made of?  
  (product, channel, coverage composition)

- How is premium distributed?  
  (mean vs median, dispersion, tails, concentration)

- Where does modelling effort matter most financially?

### Key outputs

✔ Portfolio mix diagnostics (product × channel × coverage)  
✔ Premium dispersion and concentration analysis  
✔ Tail contribution (top 1%, 5%, 10% of policies)  
✔ Coverage → severity tail validation (P90 / P95 / P99)  
✔ Explicit pricing design note (intentional weak risk differentiation)  
✔ Leadership framing and portfolio steering implications  

---

## Phase 3 — Loss Ratio Drill-Down (Actuarial View) (v0.3)

Phase 3 introduces **actuarial loss ratio analysis** on the frozen synthetic portfolio,
building directly on the pricing context established in Phase 2.

The focus of this phase is **not model fitting** —  
it is **profitability diagnosis and decision prioritisation** using
earned premium logic and premium-weighted views.

Loss ratios are treated as **decision signals**, not just summary metrics.

### Key questions answered

- Where is the portfolio **making or losing money**?
- Which combinations of **product × channel** dominate financial risk?
- Are adverse loss ratios driven by **frequency, severity, or exposure mix**?
- Where would pricing, underwriting, or reinsurance review have the highest impact?

### Key outputs

✔ Earned premium–based loss ratio calculations  
✔ Premium-weighted aggregation (financial materiality lens)  
✔ Product × Channel loss ratio heatmap (executive view)  
✔ Clear separation of **diagnosis** vs **modelling**  
✔ Explicit decision framing for pricing and portfolio steering 

## Phase 4 — Macro & CAT Scenario Sensitivity (Board View) (v0.4)

Phase 4 extends the digital twin from *diagnosis* into **forward-looking stress testing
and executive decision support**.

This phase introduces a **scenario engine** that translates macroeconomic shocks and
catastrophe events into portfolio-level paid loss impacts — both **gross** and **net of reinsurance**.

The focus is **not forecasting**.
It is understanding *exposure, sensitivity, and protection effectiveness* under stress.

---

### Key questions answered

- How sensitive is the portfolio to **inflation, repair costs, unemployment, and CAT events**?
- Which scenarios produce **material paid-loss uplift**?
- Which products **drive the stress impact**?
- How much risk is **absorbed by reinsurance**, and how much remains net?
- Where does residual risk concentrate *after* QS + XL protection?

---

### Key outputs

✔ Scenario engine driven by macro and CAT shocks  
✔ Portfolio-level paid loss impact (gross view)  
✔ Product-level attribution of scenario uplift  
✔ Bootstrap uncertainty bands for key stresses  
✔ **Reinsurance effectiveness analysis (QS + XL)**  
✔ Executive-ready board packs (PPT)  
✔ **Interactive Streamlit scenario simulator** for live decision exploration  

---

### Board artefacts

- `04_board_scenario_pack.pptx`  
  *Gross portfolio impact under macro & CAT scenarios*

- `04_board_scenario_pack_with_RI.pptx`  
  *Gross → Net view with QS + XL reinsurance protection*

These decks are structured for **pricing, underwriting, and reinsurance committees**,
not exploratory analysis.

---

### Executive simulator (Streamlit)

This phase also introduces an **interactive executive scenario tool**.

Features:
- Live sliders for:
  - Inflation shock
  - Repair cost shock
  - Unemployment shift
  - CAT year override
- Reinsurance structure controls:
  - Quota share %
  - XL retention
  - XL limit
- Instant visibility of:
  - Gross paid loss
  - Net paid loss after RI
  - Risk transfer efficiency
  - Product-level attribution

Location:
- `notebooks/ui/scenario_simulator_exec_demo.py`

Purpose:
> Convert static board analysis into a **live decision conversation**.

---

## Phase 5 — Anomaly Audit & Model Robustness (Modelling Readiness Gate) (v0.5)

Phase 5 introduces a formal modelling-readiness certification layer on top of the frozen synthetic portfolio.

This phase does not fit predictive models.

Instead, it validates that the portfolio is structurally and statistically ready
for controlled frequency modelling in the next phase.

In regulated insurance environments, predictive modelling does not begin
until exposure integrity, anomaly bounds, and distributional assumptions
have been formally validated.

Phase 5 mirrors that discipline.

---

### Key questions answered

- Is exposure derived correctly and free from structural inconsistencies?
- Are anomalies rare, bounded, and explainable?
- Does claim count exhibit statistically significant overdispersion?
- Is Negative Binomial GLM justified over Poisson?
- Is meaningful risk signal present across rating factors?
- Is temporal leakage prevented before model fitting?

---

### Key outputs

✔ Exposure derivation from policy start and end dates  
✔ Detection and controlled handling of non-positive exposure cases (~0.1%)  
✔ Poisson dispersion testing (Pearson χ² / dof ≈ 88)  
✔ Formal justification for Negative Binomial frequency modelling  
✔ Risk signal validation across rating dimensions (e.g. vehicle_age)  
✔ Fraud-like structural clustering diagnostics  
✔ Temporal train/test split to prevent forward-looking bias  
✔ Phase 5 modelling-readiness certification  

---

### Why this matters

Before fitting a GLM, pricing teams must ensure:

- exposure is structurally consistent
- statistical assumptions are defensible
- risk differentiation exists in the data
- modelling pipelines are leakage-safe

Phase 5 ensures that the portfolio is not only analytically interesting,
but statistically and procedurally ready for predictive modelling.

---

## Phase 6 — Technical Frequency Model (Negative Binomial GLM) (v0.6)

Phase 6 introduces the first predictive modelling layer within the governed digital twin architecture.

This phase implements a Negative Binomial (NB2) claim frequency model, transitioning the project from modelling-readiness certification (Phase 5) into structured actuarial modelling.

The objective is not simply to fit a model.

It is to recover structured risk signal in a way that is:

- statistically defensible
- leakage-safe
- commercially interpretable
- deployable as rating factors

This phase mirrors how regulated pricing teams formally introduce predictive modelling into a governed environment.

---

### Key questions answered

- Is Negative Binomial statistically justified over Poisson?
- Does the portfolio exhibit recoverable and stable risk differentiation?
- Can claim frequency be modelled per policy-year using exposure offsets?
- Is fraud correctly separated from the technical pricing base?
- Does the model demonstrate out-of-sample calibration and lift?
- Are rating relativities implementable and stable?

---

### Key outputs

✔ Negative Binomial GLM with log(exposure) offset  
✔ Formal overdispersion validation (Poisson vs NB comparison)  
✔ Fraud excluded from technical frequency base  
✔ Temporal train/test split to prevent leakage  
✔ Decile calibration and ~4.3× lift validation  
✔ Vehicle age banding with monotonicity checks  
✔ Structured pricing relativities (exp(beta))  
✔ Exported deployment-ready artefacts  

---

### Statistical validation

The portfolio exhibits statistically significant overdispersion, formally justifying the NB2 distribution.

The model demonstrates:
- Strong out-of-sample risk separation
- Stable predicted means across train/test
- Calibration consistency across deciles
- Economically interpretable segmentation
Primary frequency drivers:
- Product type (dominant differentiator)
- Channel
- Vehicle age bands

---

### Export artefacts

Phase 6 produces structured outputs under: `outputs/phase6/`

- `relativities_product.csv`
- `relativities_channel.csv`
- `relativities_vehicle_age_band.csv`
- `phase6_exec_metrics.json`

These artefacts mirror internal pricing workflows, where modelling outputs are converted into rating-engine-ready factors rather than remaining notebook-bound.

---

### Why this matters

Phase 6 marks the transition from:

**Modelling readiness → Certified predictive pricing layer**

The digital twin now contains:
- A distributionally justified frequency model
- Governance-aligned exposure specification
- Leakage-safe validation structure
- Deployable rating relativities
- All built on the frozen and validated dataset from Phase 1.

---

## Phase 7 — Fraud Overlay Architecture (Lift + Ring Detection + SIU Decisioning) (v0.7)

Phase 7 introduces the fraud analytics control layer within the Insurance Portfolio Digital Twin.

This phase does not treat fraud as a standalone classification exercise.

Instead, it implements a governance-aligned fraud overlay architecture that mirrors how regulated insurers operationalise fraud detection alongside pricing.

The objective is not simply to maximise AUC.

It is to:

- detect structured fraud signal
- separate pricing from fraud risk
- optimise review decisions economically
- validate structural robustness
- introduce monitoring discipline

This phase formally introduces a fraud control layer on top of the pricing architecture established in Phase 6.

---

### Key questions answered

- Can fraud propensity be modelled in a leakage-safe manner?
- Does the portfolio exhibit recoverable fraud signal?
- Is structural ring behaviour detectable and economically meaningful?
- How much fraud value can SIU capture under capacity constraints?
- What review threshold maximises net economic benefit?
- Is the fraud model stable and explainable under governance standards?

---

### Key components introduced

#### 1️⃣ Fraud Propensity Model

✔ Regularised Logistic Regression  
✔ Temporal holdout validation  
✔ Cross-validated isotonic calibration  
✔ ROC-AUC ≈ 0.82  
✔ PR-AUC validation for imbalanced data  
✔ Baseline lift ≈ 39× (top vs bottom decile)

Fraud is modelled independently from the technical pricing layer, preserving structural separation between pricing risk and fraud risk.

---

#### 2️⃣ Structural Ring Detection & Overlay

✔ Ring-level clustering diagnostics  
✔ Fraud ratio estimation at ring level  
✔ Overlay integration (propensity + ring signal)  
✔ Overlay lift ≈ 134×  
✔ Shuffle validation (overlay robustness test)

This confirms that overlay performance is structural rather than accidental or leakage-driven.

---

#### 3️⃣ Cost-Weighted Triage (SIU Decision Layer)

Fraud scoring is extended into operational decisioning.

Introduced:

✔ Expected fraud loss estimation (paid + reserve proxy)  
✔ Review cost incorporation  
✔ Capacity simulation (review top X%)  
✔ Net economic value estimation  
✔ Threshold optimisation via EVR (Expected Value of Review)

This transforms fraud modelling from predictive ranking into capital allocation logic.

---

#### 4️⃣ EVR Sensitivity Analysis

✔ Recovery-rate sensitivity testing  
✔ Net value variation across assumptions  
✔ Robustness validation of threshold policy

Ensures that review strategy remains economically defensible under varying recovery conditions.

---

#### 5️⃣ Drift Monitoring & Alerts

✔ PSI-based fraud score monitoring  
✔ Governance thresholds (warning / critical)  
✔ Retraining trigger signalling

Establishes first-pass model monitoring discipline consistent with production environments.

---

#### 6️⃣ Interpretability Pack (SHAP)

✔ Global SHAP feature importance  
✔ Post-encoding feature name preservation  
✔ Ranked importance export

Supports model risk governance and executive transparency.

---

### Export artefacts

Phase 7 produces structured outputs under: `outputs/phase7/`

- `phase7_claim_scores_test.csv`
- `phase7_ring_summary.csv`
- `phase7_siu_capacity_table.csv`
- `phase7_siu_cost_capacity_table.csv`
- `phase7_siu_threshold_policy_table.csv`
- `phase7_exec_metrics.json`

These artefacts mirror internal fraud governance workflows where scoring outputs are translated into operational policy tables.

---

### Why this matters

In regulated insurance environments, fraud modelling must answer:

- Who should be reviewed?
- How many claims can be reviewed?
- What is the net economic value?
- Is the signal structurally defensible?
- Is the model stable under drift?
- Is it explainable to governance committees?

Phase 7 embeds fraud detection as a controlled overlay within the digital twin architecture.

This marks the transition from:

**Predictive modelling → Operational fraud control framework**

---

### Notebooks

- `00_data_gen_validation.ipynb` — generator sanity checks & governance gates  
- `01_eda_frozen_synthetic_universe.ipynb` — actuarial realism validation  
- `02_portfolio_mix_premium_pricing_context.ipynb` — pricing context on frozen data
- `03_loss_ratio_drilldown_actuarial.ipynb` —  Actuarial loss ratio drill-down using earned premium logic, with
  executive-ready visualisation and governance anchoring.
- `04_macro_cat_sensitivity.ipynb` — Scenario engine, macro sensitivities, CAT stress, uncertainty bands, and RI impact
- `05_anomaly_audit_and_model_robustness.ipynb` — Exposure validation, anomaly diagnostics, dispersion testing, risk signal stability checks, and modelling-readiness certification.
- `06_frequency_model_nb_glm_risk_signal_recovery.ipynb` — Negative Binomial frequency model with exposure offset, calibration, lift validation, monotonicity audit, and pricing-ready relativities export.
- `07_fraud_model_overlay_and_ring_detection.ipynb` —  
  Fraud propensity modelling, cross-validated calibration, lift analysis,
  ring overlay integration, EVR optimisation, SIU capacity simulation,
  drift monitoring, SHAP interpretability, and governance export pack.

All built on **frozen, governed data** from Phase 1.

---

## Repository structure

insurance-digital-twin/

├── data_gen/

│ ├── config.py # Portfolio assumptions & targets

│ ├── generators.py # Synthetic data generation logic

│ ├── schemas.py # Entity schemas (documentation & typing)

│ └── cli.py # Dataset generation + freeze entry point

│

├── data/

│ └── raw/

│ └── dataset_manifest.json # Dataset hashes + metadata

├── notebooks/

│ ├── 00_data_gen_validation.ipynb

│ ├── 01_eda_frozen_synthetic_universe.ipynb

│ ├── 02_portfolio_mix_premium_pricing_context.ipynb

│ └── 03_loss_ratio_drilldown_actuarial.ipynb

│ ├── 04_macro_cat_sensitivity.ipynb  

│ └── ui/  

│     └── scenario_simulator_exec_demo.py  

│ └── 05_anomaly_audit_and_model_robustness.ipynb

│ └── 06_frequency_model_nb_glm_risk_signal_recovery.ipynb

│ └── 07_fraud_model_overlay_and_ring_detection.ipynb

└── README.md


---

## How to run

### Phase 1 — Generate & freeze dataset

```bash
python -m data_gen.cli

```

This produces a **frozen dataset** with a versioned manifest and cryptographic hashes.

**Phase 2 — Pricing context analysis**

Open and run:

- notebooks/02_portfolio_mix_premium_pricing_context.ipynb

**Phase 3 — Loss Ratio Drill-Down**

Open and run:

- notebooks/03_loss_ratio_drilldown_actuarial.ipynb

**Phase 4 — Macro & CAT Scenario Sensitivity (Board View)**

Open and run:

- notebooks/04_macro_cat_sensitivity.ipynb

(Optional interactive demo)

- notebooks/ui/scenario_simulator_exec_demo.py

They consume the frozen outputs from Phase 1.

**Phase 5 - Anomaly Audit & Model Robustness (Modelling Readiness Gate)**

Open and run:

- notebooks/05_anomaly_audit_and_model_robustness.ipynb

**Phase 6 - Technical Frequency Model (NB GLM)**

Open and run:

- notebooks/06_frequency_model_nb_glm_risk_signal_recovery.ipynb

**Phase 6 - Fraud Overlay Architecture (Lift + Ring Detection + SIU Decisioning)**

Open and run:

- notebooks/07_fraud_model_overlay_and_ring_detection.ipynb

⚠️ Phase 2, 3, 4, 5, 6 & 7 **do not regenerate data.**
---

## Releases

- **v0.1 — Dataset Freeze & Governance**
- **v0.2 — Portfolio Mix & Pricing Context**
- **v0.3 — Loss Ratio Drill-Down**
- **v0.4 — Macro & CAT Sensitivity**
- **v0.5 — Anomaly Audit & Modelling Readiness**
- **v0.6 — Technical Frequency Model (NB GLM)**
- **v0.7 — Fraud Overlay Architecture (Lift + Ring Detection + SIU Decisioning)**

---

## **What’s next**

**v0.8 — Scenario Simulator (Inflation Shock + Capital Stress Layer)**

- Inflation shock sliders  
- Frequency + severity stress propagation  
- Fraud-adjusted stress view  
- Gross → net capital translation  
- Executive auto-summary integration
