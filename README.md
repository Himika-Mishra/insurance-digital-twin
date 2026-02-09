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

### Notebooks

- `00_data_gen_validation.ipynb` — generator sanity checks & governance gates  
- `01_eda_frozen_synthetic_universe.ipynb` — actuarial realism validation  
- `02_portfolio_mix_premium_pricing_context.ipynb` — pricing context on frozen data
- `03_loss_ratio_drilldown_actuarial.ipynb` —  Actuarial loss ratio drill-down using earned premium logic, with
  executive-ready visualisation and governance anchoring.
- `04_macro_cat_sensitivity.ipynb` —  
  Scenario engine, macro sensitivities, CAT stress, uncertainty bands, and RI impact

### Why this matters

At board level, the question is rarely:
> “What is the loss ratio?”

It is:
> **What happens under stress, where does the money go, and how protected are we?**

Phase 4 connects:
- actuarial structure
- macro stress
- reinsurance economics
- executive decision framing

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

│

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

⚠️ Phase 2 & 3 **does not regenerate data.**

Open and run:

- notebooks/03_loss_ratio_drilldown_actuarial.ipynb

**Phase 4 — Macro & CAT Scenario Sensitivity (Board View)**

Open and run:

- notebooks/04_macro_cat_sensitivity.ipynb

(Optional interactive demo)

- notebooks/ui/scenario_simulator_exec_demo.py

⚠️ Phase 2, 3 & 4 **do not regenerate data.**

They consume the frozen outputs from Phase 1.

---

## **Releases:**  

- **v0.1 — Dataset Freeze & Governance**  
  Frozen synthetic insurance universe with validation gates and manifest.

- **v0.2 — Portfolio Mix & Premium Distributions (Pricing Context)**  
  Pricing context analysis on frozen data: mix, dispersion, concentration, and steering insights.

- **v0.3 — Loss Ratio Drill-Down (Actuarial View)**  
  Earned premium–based loss ratio analysis with product × channel drill-down
  and executive-ready visualisation.

- **v0.4 — Macro & CAT Scenario Sensitivity (Board View)**  
  Scenario engine, board packs, reinsurance effectiveness, and executive simulator.

---

## **What’s next**

**v0.5 — Anomaly Audit & Model Robustness**

Controlled “real-world messiness” diagnostics:
- negative paid
- zero premiums
- late repudiations
- process artefacts

Ensuring downstream models remain:
- stable
- explainable
- production-safe
