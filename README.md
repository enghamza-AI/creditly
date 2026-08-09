# creditly

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-from--scratch-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/pandas-data%20wrangling-150458?style=flat&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/pytest-tested-0A9EDC?style=flat&logo=pytest&logoColor=white" alt="pytest">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="License">
</p>

> A loan-default risk predictor built by fusing **Lending Club** loan-level data with **U.S. Census ACS** income-by-ZIP data — two sources that don't share a clean key, cleaned, merged, and modeled from first principles (no `sklearn` shortcuts).



---

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [The Problem, in Plain English](#the-problem-in-plain-english)
- [Datasets](#datasets)
- [What This Project Teaches](#what-this-project-teaches)
- [Project Roadmap](#project-roadmap)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [Design Decisions & Notes](#design-decisions--notes)
- [License](#license)

---

## Why This Project Exists

Most tutorials hand us a single clean CSV and a `.fit()` call. Real-world ML almost never looks like that. This project is deliberately harder in a specific way: **two independent, messy, real datasets that must be reconciled before any modeling can happen** — which is closer to what ML work actually looks like in industry.

The goal isn't just "get a model that predicts default." The goal is to build the *judgment* to reason about data quality, missingness, and leakage — the stuff that separates someone who calls `.fillna(0)` reflexively from someone who understands why the data is missing in the first place.

And I am not here to be called as someone who calls fillna without any reasoning at all - 

## The Problem, in Plain English

A lender wants to estimate, **before approving a loan**, how likely a borrower is to default. Lending Club provides historical loan records (loan amount, interest rate, borrower income, employment length, and — critically — whether the loan was ultimately paid off or charged off). On their own, these records say something about the *borrower*, but nothing about the *economic context* they live in.

Census ACS data adds that context: median household income, income distribution, and other socioeconomic indicators, aggregated by ZIP-adjacent geography. Joining the two lets a model ask questions like: *does a borrower's income relative to their local area matter more than their income in isolation?*

**Success for this project** is not "high accuracy." It's:
1. A merge that is *honest* about its limitations 
2. A missing-data strategy that's justified by reasoning, not convenience.
3. A logistic regression built from raw NumPy that you can explain end-to-end — normal equation, gradient descent, and why log-loss (not MSE) is the right objective.
4. Code organized as functions/modules with tests, not a 300-cell notebook.

## Datasets

| Dataset | Source | Grain | Role |
|---|---|---|---|
| Lending Club loan data | [Kaggle: wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | One row per loan | Target variable (default/paid) + borrower features |
| Census ACS income by ZIP | [data.census.gov](https://data.census.gov/) (ACS 5-Year Estimates) | One row per ZCTA (ZIP Code Tabulation Area) | Local income context feature(s) |

See [Dataset Download Instructions](#setup) below for exact steps.

## What I learned from this project

- **Missing data reasoning**: MCAR / MAR / MNAR — diagnosing *why* data is missing before deciding *how* to handle it.
- **Data joining under real-world constraints**: reconciling two datasets with no shared clean key, including privacy-masked geography.
- **Categorical encoding**: one-hot vs. target encoding, and precisely how/why target encoding leaks information if computed before the train/test split.
- **Linear regression from first principles**: the normal equation (closed-form) and gradient descent, implemented in raw NumPy.
- **Logistic regression from first principles**: why log-loss (cross-entropy) is the correct loss function for classification instead of MSE, derived rather than asserted.
- **Software fundamentals for ML code**: functions over notebook cells, a `src/` package layout, and `pytest` tests for data-transformation logic.

## Project Roadmap

| Day | Focus |
|---|---|
| **Day 1** | Load both datasets, profile them, understand missingness, build a working (if imperfect) merge |
| **Day 2** | Resolve missing data (MCAR/MAR/MNAR-informed), encode categoricals correctly (post-split), feature engineering |
| **Day 3** | Linear regression from scratch (normal equation + gradient descent) as a warm-up, then logistic regression from scratch (gradient descent + log-loss), evaluation |

## Repository Structure

See full structure in [Project Setup — Folder Structure](#-2-folder-structure) of the accompanying planning notes, or the `tree` below:

```
creditly/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/creditly/
│   ├── data/
│   ├── features/
│   └── models/
├── tests/
├── reports/figures/
└── scripts/
```

## Setup

```bash
git clone https://github.com/enghamza-AI/creditly.git
cd creditly
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dataset download instructions live in the Day 1 notes (`notebooks/day1_eda_and_merge.ipynb` header cell) and in the project write-up.

## Usage

```bash
python scripts/run_pipeline.py
```

_(Day 1: loads, profiles, and merges the raw data. Later days extend this.)_

## Testing

```bash
pytest tests/ -v
```

## Design Decisions & Notes

- **Lending Club ZIP codes are privacy-masked** to a 3-digit prefix (e.g. `112xx`), not a full 5-digit ZIP. This is the central data-engineering constraint of the whole project and is documented in detail in the Day 1 notes — it changes how the Census side must be aggregated before any join is valid.
- Target encoding is computed **only on the training fold**, after the train/test split, to avoid leakage — this is a deliberate, tested constraint, not an oversight.
- Models are implemented in raw NumPy for Days 1–3. `sklearn` is intentionally deferred until the fundamentals are demonstrably understood.

## License

MIT — see [LICENSE](LICENSE).
