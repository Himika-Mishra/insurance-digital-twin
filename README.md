# insurance-digital-twin
Synthetic personal-lines insurance portfolio built as a governed digital twin, with dataset freezing, validation gates, and actuarial realism.
=======
# Insurance Portfolio Digital Twin  
### Phase 1 — Synthetic Insurance Universe with Governance

This repository contains **Phase 1** of a multi-stage *Insurance Portfolio Digital Twin*.

The focus of this phase is **not modelling** —  
it is **data generation, governance, validation, and auditability**.

Before pricing, fraud, forecasting, or scenario analysis can be trusted,  
the underlying dataset must be **frozen, reproducible, and defensible**.

That is what this phase delivers.

---

## Why this project exists

In real insurance environments, analytical credibility depends on:
- reproducibility
- traceability
- controlled imperfections
- governance before modelling

Most public analytics projects skip these steps.

This project does not.

---

## Phase 1 scope (current)

**Delivered in this repository:**

✔ Synthetic personal-lines insurance universe  
✔ Policyholders, policies, claims, macro environment  
✔ Explicit modelling assumptions (documented in config)  
✔ Controlled anomaly injection (real-world messiness)  
✔ Validation gates (actuarial sanity checks)  
✔ Dataset freeze with manifest and cryptographic hashes  
✔ Auditable, versioned data artefact  

**Explicitly not included yet:**
- pricing models
- fraud models
- scenario simulators
- dashboards or UI

These will be added incrementally in later phases.

---

## What makes this different

This repository treats synthetic data as a **governed asset**, not a toy dataset.

It includes:
- deterministic generation via fixed random seeds
- hash-based dataset locking
- validation checks aligned to actuarial practice
- anomaly rates that are intentional, rare, and bounded

This mirrors how internal insurance analytics platforms are built.

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

## How to run (Phase 1)

```bash
python -m data_gen.cli

