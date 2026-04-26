# From Bias to Fair: Can Synthetic Data Bridge the Gap?

Evaluation of the impact of synthetic data on algorithmic bias mitigation in machine learning models [CatBoost, Logistic Regression, SVM], focusing on datasets [COMPAS, Adult, Diabetes 130-US Hospitals for Years 1999-2008] and generators [CTGAN, Gaussian Copula, TVAE, TabDDM].

---

## Experimental Scenarios

The experiments are organized into six scenarios that allow isolating and comparing the effect of different intervention strategies on bias and predictive performance.

---

### S1 — Original (baseline without intervention)

Training and evaluating models on the original data, without any mitigation pre-processing or synthetic data generation. Serves as a baseline for comparison with all other scenarios.

---

### S2 — Pre-processing Mitigation

Application of three classic bias mitigation methods in the data pre-processing stage:

| Method | Description |
|--------|-----------|
| **Reweighing** | Assigns different weights to instances to compensate for disparities between groups |
| **DIRemover** (Disparate Impact Remover) | Transforms features to reduce disparate impact while maintaining predictive utility |
| **LFR** (Learning Fair Representations) | Learns latent representations that satisfy fairness constraints |

---

### S3 — Synthetic Data on Entire Dataset

Synthetic data generation applied to the entire dataset, **without generation condition** and **without conditional mitigation intent**. Works as an experimental control to distinguish the data augmentation effect from any deliberate mitigation effect.

The proportion of added synthetic data is controlled by the scale parameter **α**:

$$\alpha \in \{1.5,\ 2,\ 3\}$$

| α | Description |
|---|-----------|
| 1.5 | Increases the dataset by 50% with synthetic data |
| 2 | Doubles the dataset size |
| 3 | Triples the dataset size |

---

### S4 — Synthetic Oversampling of Minority Groups

Synthetic data generation targeted at **minority groups**, based on the distributions of the majority group, aiming to **equalize proportions** between the sensitive attribute (race) and the outcome.

The intervention seeks to balance racial representation without altering the majority group's distribution.

---

### S5 — Synthetic Data by Subgroup (outcome × race)

Synthetic data generation **exclusively for the underrepresented group** in the outcome, using cross-separation between sensitive attribute and outcome:

| Subgroup | Description |
|----------|-----------|
| `african-american + yes` | African-American individuals with positive outcome |
| `african-american + no` | African-American individuals with negative outcome |
| `caucasian + yes` | Caucasian individuals with positive outcome |
| `caucasian + no` | Caucasian individuals with negative outcome |

For each combination, synthetic data is generated for the **underrepresented class**, correcting specific imbalances within each race × outcome stratum.

---

### S6 — Hybrid Mitigation (Pre-processing + Synthetic Data)

Combination of the **Reweighing** pre-processing mitigation method with the synthetic generation strategy of **S5** (synthetic data focused on the underrepresented group in the outcome).

Allows evaluating the synergistic or complementary effect between weight correction and targeted subgroup augmentation.

## Experimental Setup

- 10 seeds
- Z-score normalization
- Train/test split of 80/20
- Fit / transform / only train dataset

---

## How to Run

### 1. Running the Experiment Pipeline
The main experiment script generates synthetic data, applies mitigators, trains models, and computes metrics. You can run it for one or multiple datasets sequentially:

```bash
# Run for a single dataset (e.g., adult)
python scripts/run_experiment.py --dataset adult

# Run for multiple datasets in sequence
python scripts/run_experiment.py --dataset compas adult diabetes

# To see all options
python scripts/run_experiment.py --help
```

### 2. Generating Plots
All plotting scripts support processing multiple datasets in a single execution. The plots are saved automatically under `plots/<dataset_name>/...`.

**Paper Plots (Trade-offs & Fairness vs Utility)**
```bash
python scripts/plots/generate_paper_plots.py --dataset compas adult diabetes
```

**Data Analysis & Fairness Overlaps (Bhattacharyya)**
```bash
python scripts/plots/plot_data_analysis_fairness.py --dataset compas adult diabetes
```

**Individual Synthetic Fidelity (Distributions & Correlations)**
```bash
python scripts/plots/plot_individual_synthetic_fidelity.py --dataset compas adult diabetes
```

**Utility Metrics (ROC, PR Curves & Accuracy Trends)**
```bash
python scripts/plots/plot_utility_metrics.py --dataset compas adult diabetes
```

**SDV Fidelity Heatmaps & KDE Overlays (JSD Analysis)**
```bash
# You can also specify the reference scenario for the JSD evaluation
python scripts/plots/plot_sdv_fidelity.py --dataset compas adult diabetes --scenario S3_2.0
```