# Insurance Portfolio Digital Twin

Synthetic personal-lines insurance portfolio built as a **governed digital twin**,  
with dataset freezing, validation gates, and actuarial realism.

This repository evolves in **phases**, each adding analytical depth while preserving
governance, reproducibility, and auditability.

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

### Notebooks

- `00_data_gen_validation.ipynb` — generator sanity checks & governance gates  
- `01_eda_frozen_synthetic_universe.ipynb` — actuarial realism validation  
- `02_portfolio_mix_premium_pricing_context.ipynb` — pricing context on frozen data  

This phase establishes a **defensible baseline** for:
- loss ratio drill-downs
- actuarial GLM / NB modelling
- pricing uplift estimation
- fraud and scenario modelling

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

│
├── notebooks/

│ ├── 00_data_gen_validation.ipynb

│ └── 01_eda_frozen_synthetic_universe.ipynb

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

⚠️ Phase 2 **does not regenerate data.**

It consumes the frozen outputs from Phase 1.

---

## **Releases:**  

- **v0.1 — Dataset Freeze & Governance**

  Frozen synthetic insurance universe with validation gates and manifest.

- **v0.2 — Portfolio Mix & Premium Distributions (Pricing Context)**

  Pricing context analysis on frozen data: mix, dispersion, concentration, and steering insights.

---

## **What’s next**

**v0.3 — Loss Ratio Drill-Down (Actuarial View)**

Frequency vs severity decomposition by product × channel × risk, using earned premium logic.

