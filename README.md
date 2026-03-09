# Insurance Portfolio Digital Twin

Synthetic personal-lines insurance portfolio built as a **governed digital twin**,  
with dataset freezing, validation gates, and actuarial realism.

This repository evolves in **phases**, each adding analytical depth while preserving  
governance, reproducibility, and auditability.

Designed to mirror how **regulated insurance analytics platforms** are built internally,  
rather than how public modelling demos are typically presented.

---

## Key Capabilities

This project implements an end-to-end **insurance portfolio analytics stack**, including:

- Synthetic portfolio generation with governance controls
- Actuarial loss ratio diagnostics
- Negative Binomial frequency modelling
- Fraud detection overlay with operational triage optimisation
- Macro & catastrophe scenario stress testing
- Stochastic portfolio loss simulation
- Capital stress modelling (99.5% solvency proxy)
- Catastrophe-style exceedance probability (EP) curves
- Pricing response simulation with demand elasticity

The system is designed to mimic how **pricing, fraud, and risk analytics operate together inside an insurer**.

---

## Digital Twin Architecture

The project evolves through a governed analytics pipeline:

Synthetic Portfolio  
↓  
Governance & Dataset Freeze  
↓  
Portfolio Diagnostics  
↓  
Actuarial Loss Ratio Analysis  
↓  
Frequency Modelling (NB GLM)  
↓  
Fraud Detection Overlay  
↓  
Scenario Stress Engine  
↓  
Stochastic Portfolio Simulation  
↓  
Capital & Tail Risk Analysis  
↓  
Pricing Strategy Simulation

---

# Project Phases

This project is structured as a **multi-phase insurance analytics build**, where each  
phase produces a **stable, defensible artefact** before moving forward.

---

# Phase 1 — Synthetic Insurance Universe & Governance (v0.1)

The focus of Phase 1 is **not modelling** — it is **data generation, governance, validation, and auditability**.

Before pricing, fraud, forecasting, or scenario analysis can be trusted, the underlying dataset must be **frozen, reproducible, and defensible**.

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

**Delivered in this repository**

- Synthetic personal-lines insurance universe  
- Policyholders, policies, claims, macro environment  
- Explicit modelling assumptions (`config.py`)  
- Controlled anomaly injection  
- Validation gates (actuarial sanity checks)  
- Dataset freeze with cryptographic manifest  
- Auditable versioned dataset artefact  

**Explicitly not included yet**

- pricing models  
- fraud models  
- scenario simulators  
- dashboards  

These are introduced progressively in later phases.

---

# Phase 2 — Portfolio Mix & Premium Distributions (Pricing Context) (v0.2)

Phase 2 builds **pricing context** on top of the frozen dataset produced in Phase 1.

No data is regenerated or modified.

### Key questions

- What is the portfolio composition?
- Where is premium concentrated?
- Which segments financially dominate the book?

### Key outputs

- Portfolio mix diagnostics  
- Premium dispersion analysis  
- Tail contribution analysis (top 1%, 5%, 10%)  
- Coverage severity validation (P90/P95/P99)  
- Pricing design note  

---

# Phase 3 — Loss Ratio Drill-Down (Actuarial View) (v0.3)

Phase 3 introduces **earned premium–based loss ratio diagnostics**.

The focus is **portfolio profitability diagnosis**, not predictive modelling.

### Key outputs

- Earned premium loss ratio calculations  
- Premium-weighted aggregation  
- Product × channel loss ratio heatmap  
- Financial materiality prioritisation  

---

# Phase 4 — Macro & CAT Scenario Sensitivity (Board View) (v0.4)

Phase 4 extends the digital twin into **forward-looking stress testing**.

Introduces a **scenario engine** translating macro shocks into paid-loss impacts.

### Key outputs

- Macro scenario engine  
- CAT stress layer  
- Portfolio gross vs net impact  
- Reinsurance protection analysis  
- Bootstrap uncertainty bands  
- Executive board packs  
- Interactive Streamlit scenario simulator  

---

# Phase 5 — Anomaly Audit & Model Robustness (Modelling Readiness Gate) (v0.5)

Phase 5 introduces **formal modelling-readiness certification**.

Predictive models are not fitted yet — instead the dataset is audited.

### Key outputs

- Exposure validation  
- Controlled anomaly detection  
- Poisson dispersion testing  
- NB modelling justification  
- Risk signal validation  
- Temporal leakage protection  

---

# Phase 6 — Technical Frequency Model (Negative Binomial GLM) (v0.6)

Phase 6 introduces the **first predictive model** in the governed environment.

A **Negative Binomial GLM** is fitted for claim frequency.

### Key outputs

- NB GLM with exposure offset  
- Poisson vs NB overdispersion validation  
- Temporal holdout validation  
- Decile calibration and lift analysis  
- Vehicle age monotonicity checks  
- Pricing relativities export  

### Export artefacts

Phase 6 produces structured outputs under: `outputs/phase6/`

- `relativities_product.csv`
- `relativities_channel.csv`
- `relativities_vehicle_age_band.csv`
- `phase6_exec_metrics.json`

These artefacts mirror internal pricing workflows, where modelling outputs are converted into rating-engine-ready factors rather than remaining notebook-bound.

---

# Phase 7 — Fraud Overlay Architecture (v0.7)

Phase 7 introduces the **fraud detection control layer**.

Fraud modelling is treated as a **governance overlay**, not a standalone classifier.

### Key components

### Fraud Propensity Model

- Logistic regression  
- Temporal holdout validation  
- Isotonic calibration  
- ROC-AUC ≈ 0.82  

### Structural Ring Detection

- Fraud clustering diagnostics  
- Overlay integration (propensity + ring signal)  
- Overlay lift ≈ 134×  
- Shuffle validation  

### Operational Decision Layer

- SIU review optimisation  
- Expected Value of Review (EVR)  
- Capacity simulations  

### Monitoring

- PSI drift monitoring  
- Governance thresholds  
- SHAP interpretability pack  

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

# Phase 8 — Portfolio Scenario Simulator & Capital Stress Engine (v0.8)

Phase 8 moves the Digital Twin beyond modelling into **portfolio decision simulation**.

Instead of predicting claims, the system now evaluates:

**How the entire insurance portfolio behaves under stress.**

This phase introduces **stochastic portfolio risk simulation**, capital stress metrics, and executive-level risk reporting.

---

## Scenario Simulation Engine

A scenario framework translates macro and portfolio shocks into portfolio outcomes.

Simulated stresses include:

- claim frequency shocks  
- severity inflation shocks  
- fraud overlays  
- catastrophe-style tail events  

---

## Stochastic Portfolio Loss Simulation

Phase 8 introduces a **Monte Carlo collective risk model**.

This produces a full **portfolio loss distribution** rather than point estimates.

Outputs include:

- expected loss  
- percentile tail losses  
- scenario loss distributions  

---

## Catastrophe Simulation Layer

Low-probability, high-severity shocks are introduced to shape the **right tail of portfolio risk**.

This better reflects how insurers evaluate capital exposure under extreme scenarios.

---

## Capital Stress Metrics

Capital stress metrics now include:

- Expected loss  
- 99% portfolio loss  
- **99.5% Solvency-style tail loss**  
- Capital required proxy  
- Solvency ratio proxy  

---

## Exceedance Probability (EP) Curve

Phase 8 introduces **catastrophe-style exceedance curves** used in Solvency II and Lloyd’s risk modelling.

Outputs include:

- probability exceedance curve  
- **return period EP curve (1-in-100, 1-in-200)**  
- tail risk diagnostics  

---

## Pricing Response Simulation

A pricing simulation layer evaluates how rate changes affect:

- portfolio loss ratios  
- premium volume  
- profitability  

Customer behaviour is incorporated using **pricing elasticity assumptions**.

---

## Portfolio Strategy Simulation

The Digital Twin can simulate how strategic actions affect portfolio economics.

Examples:

- targeted rate increases  
- fraud mitigation impact  
- capital stress mitigation  

---

## Executive Risk Dashboard

Phase 8 produces board-level artefacts summarising portfolio risk:

- loss distribution  
- inflation sensitivity  
- capital adequacy  
- EP curve tail risk  

These outputs mirror how risk analytics is presented to:

- pricing committees  
- risk committees  
- executive leadership  

---

### Export artefacts

Phase 8 produces structured outputs under: `outputs/phase8/`

- `scenario_summary.csv`
- `loss_distribution_mc.csv`
- `phase8_capital_metrics.json`
- `phase8_ep_metrics.json`
- `phase8_exec_report.txt`
- `phase8_governance_checks.json`
- `phase8_portfolio_risk_dashboard.png`
- `phase8_portfolio_risk_dashboard.png`
- `phase8_ep_curve.png`

### Example Outputs

The system produces artefacts similar to those used in insurer risk committees and pricing discussions.

Example visual outputs include:

- Portfolio loss distribution under stochastic simulation
- Inflation sensitivity analysis
- Portfolio capital adequacy diagnostics
- Catastrophe-style exceedance probability curves
- Executive portfolio risk dashboard

These outputs mirror how portfolio analytics is communicated to:

- Pricing committees
- Risk management teams
- Executive leadership

---

### Notebooks


- `00_data_gen_validation.ipynb`
- `01_eda_frozen_synthetic_universe.ipynb`
- `02_portfolio_mix_premium_pricing_context.ipynb`
- `03_loss_ratio_drilldown_actuarial.ipynb`
- `04_macro_cat_sensitivity.ipynb`
- `05_anomaly_audit_and_model_robustness.ipynb`
- `06_frequency_model_nb_glm_risk_signal_recovery.ipynb`
- `07_fraud_model_overlay_and_ring_detection.ipynb`
- `08_portfolio_scenario_simulator_capital_stress.ipynb`

Interactive simulator:

`notebooks/ui/scenario_simulator_exec_demo.py`

All analysis is built on **frozen governed data from Phase 1**.

---

# Repository Structure


insurance-digital-twin/

data_gen/
        config.py
        generators.py
        schemas.py
        cli.py

data/
raw/
    dataset_manifest.json

notebooks/
          00_data_gen_validation.ipynb
          01_eda_frozen_synthetic_universe.ipynb
          02_portfolio_mix_premium_pricing_context.ipynb
          03_loss_ratio_drilldown_actuarial.ipynb
          04_macro_cat_sensitivity.ipynb
          05_anomaly_audit_and_model_robustness.ipynb
          06_frequency_model_nb_glm_risk_signal_recovery.ipynb
          07_fraud_model_overlay_and_ring_detection.ipynb
          08_portfolio_scenario_simulator_capital_stress.ipynb

notebooks/ui/
              scenario_simulator_exec_demo.py

---

# How to Run

Generate the dataset:

```bash
python -m data_gen.cli
```bash

Then run the notebooks sequentially.

⚠️ Phase 2-8 **do not regenerate data.**
---

## Releases

- **v0.1 — Dataset Freeze & Governance**
- **v0.2 — Portfolio Mix & Pricing Context**
- **v0.3 — Loss Ratio Drill-Down**
- **v0.4 — Macro & CAT Sensitivity**
- **v0.5 — Anomaly Audit & Modelling Readiness**
- **v0.6 — Technical Frequency Model (NB GLM)**
- **v0.7 — Fraud Overlay Architecture (Lift + Ring Detection + SIU Decisioning)**
- **v0.8 — Portfolio Scenario Simulator & Capital Stress Engine**

---

## **What’s next**

**v0.9 — GenAI Executive Insight Engine**

The next phase will introduce automated executive reporting.

Capabilities will include:

- GenAI auto-generated portfolio insights 
- Natural-language scenario summaries 
- Risk interpretation for leadership audiences 
- Board-ready analytics reports 
- Executive auto-summary integration 
