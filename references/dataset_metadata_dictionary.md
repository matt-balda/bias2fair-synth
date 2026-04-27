# Dataset Metadata & Data Dictionary

This document outlines the raw dimensions, types, missingness, and the preprocessing (selection/transformation) pipelines applied to the three datasets used in this project: COMPAS, Adult, and Diabetes 130-US Hospitals.

---

## 1. COMPAS (ProPublica)

**1.1. Dimensions & Types**
*   **Original Rows & Columns:** 7,214 rows | 53 columns
*   **Final Rows & Columns:** 5,278 rows | 11 columns
*   **Data Types (Original):** 33 Categorical/Text (`object`), 16 Integer (`int64`), 4 Float (`float64`).

**1.2. Missing Data (% > 0 in Original Dataset)**
*   `violent_recid`: 100.00%
*   `vr_charge_degree`, `vr_offense_date`, `vr_case_number`, `vr_charge_desc`: 88.65%
*   `c_arrest_date`: 84.24%
*   `r_jail_in`, `r_days_from_arrest`, `r_jail_out`: 67.90%
*   `r_charge_desc`: 52.69%
*   `r_case_number`, `r_charge_degree`, `r_offense_date`: 51.89%
*   `c_offense_date`: 16.07%
*   `days_b_screening_arrest`, `c_jail_in`, `c_jail_out`: 4.26%

**1.3. Preprocessing Actions**
*   **Removed Features:** 43 features were dropped due to being highly missing (e.g., `violent_recid`, `vr_*`, `r_*`), text strings (names, case numbers), or non-predictive administrative columns.
*   **Selected Features:** `sex`, `age`, `age_cat`, `race`, `juv_fel_count`, `juv_misd_count`, `juv_other_count`, `priors_count`, `c_charge_degree`, `two_year_recid`.
*   **Transformed via `get_dummies`:** `age_cat` (e.g., "Less than 25", "25 - 45", "Greater than 45").

**1.4. Data Dictionary (Selected Features)**

| Variable | Type | Description |
| :--- | :--- | :--- |
| `two_year_recid` | Target | Binary indicator if the defendant reoffended within two years (1 = Yes, 0 = No). |
| `race` | Sensitive | Race of the defendant. Filtered and mapped to 0 (African-American) and 1 (Caucasian). |
| `sex` | Numeric | Gender of the defendant (1 = Male, 0 = Female). |
| `age` | Numeric | Age of the defendant in years. |
| `age_cat` | Categorical | Age bucket. Converted to dummy variables (`age_cat_Greater than 45`, `age_cat_Less than 25`). |
| `juv_fel_count` | Numeric | Number of prior juvenile felony convictions. |
| `juv_misd_count` | Numeric | Number of prior juvenile misdemeanor convictions. |
| `juv_other_count` | Numeric | Number of prior juvenile convictions of other types. |
| `priors_count` | Numeric | Total number of prior criminal charges. |
| `c_charge_degree` | Categorical | Degree of the current charge (mapped to 1 = Felony, 0 = Misdemeanor). |

---

## 2. Adult Income (Census)

**2.1. Dimensions & Types**
*   **Original Rows & Columns:** 48,842 rows | 15 columns
*   **Final Rows & Columns:** 48,842 rows | 55 columns (expanded due to one-hot encoding).
*   **Data Types (Original):** 9 Categorical/Text (`object`), 6 Integer (`int64`).

**2.2. Missing Data (% > 0 in Original Dataset)**
*   `occupation`: 5.75%
*   `workclass`: 5.73%
*   `native-country`: 1.75%
*(Note: Represented as "?" in the raw UCI file).*

**2.3. Preprocessing Actions**
*   **Removed Features:** 4 features dropped: `fnlwgt` (survey weight, non-predictive), `race` and `native-country` (to avoid demographic confounding with the sensitive attribute), and the original `income` text column.
*   **Selected Features:** `age`, `workclass`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`.
*   **Transformed via `get_dummies`:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`.

**2.4. Data Dictionary (Selected Features)**

| Variable | Type | Description |
| :--- | :--- | :--- |
| `income` | Target | Binary indicator if the individual earns >50K annually (1 = >50K, 0 = <=50K). |
| `sex` | Sensitive | Gender of the individual (1 = Male, 0 = Female). |
| `age` | Numeric | Age of the individual. |
| `education-num` | Numeric | Ordinal mapping of the highest level of education completed. |
| `capital-gain` | Numeric | Capital gains recorded for the individual. |
| `capital-loss` | Numeric | Capital losses recorded for the individual. |
| `hours-per-week` | Numeric | Number of hours worked per week. |
| `workclass`, `education`, `marital-status`, `occupation`, `relationship` | Categorical | Demographic and employment status categories (one-hot encoded). |

---

## 3. Diabetes 130-US Hospitals

**3.1. Dimensions & Types**
*   **Original Rows & Columns:** 101,766 rows | 48 columns
*   **Final Rows & Columns:** 95,309 rows | 40 columns
*   **Data Types (Original):** 37 Categorical/Text (`object`), 11 Integer (`int64`).

**3.2. Missing Data (% > 0 in Original Dataset)**
*   `weight`: 96.86%
*   `max_glu_serum`: 94.75% (Handled as valid "None" category)
*   `A1Cresult`: 83.28% (Handled as valid "None" category)
*   `medical_specialty`: 49.08%
*   `payer_code`: 39.56%
*   `race`: 2.23%
*   `diag_3`: 1.40%, `diag_2`: 0.35%, `diag_1`: 0.02%

**3.3. Preprocessing Actions**
*   **Removed Features:** 10 features dropped. Identifiers (`encounter_id`, `patient_nbr`), extremely high-missing features (`weight`, `payer_code`, `medical_specialty`), constant/single-value drugs (`examide`, `citoglipton`), and highly cardinal/leakage-prone ICD-9 diagnoses (`diag_1`, `diag_2`, `diag_3`).
*   **Selected Features:** `race`, `gender`, `age`, admission/discharge IDs, temporal metrics (time in hospital, procedures, lab tests), medical history (outpatient/inpatient numbers), lab tests (`max_glu_serum`, `A1Cresult`), 21 specific medications, `change`, `diabetesMed`, and `readmitted`.
*   **Transformed:** Age mapped to ordinal midpoints (e.g., [40-50) -> 45). Medications and lab results mapped to 0-3 ordinals. No direct `get_dummies` was applied; all categorical data was ordinalized mathematically.

**3.4. Data Dictionary (Selected Features)**

| Variable | Type | Description |
| :--- | :--- | :--- |
| `readmitted` | Target | Binary indicator of hospital readmission (1 = Yes [<30 or >30 days], 0 = No). |
| `race` | Sensitive | Patient's race. Filtered and mapped to 0 (African-American) and 1 (Caucasian). |
| `gender` | Numeric | Patient's gender (1 = Male, 0 = Female). |
| `age` | Numeric | Age bracket mapped to its numeric midpoint (e.g., 45, 55, 65). |
| `time_in_hospital` | Numeric | Days elapsed between admission and discharge. |
| `num_lab_procedures` | Numeric | Number of lab tests performed during the encounter. |
| `num_procedures` | Numeric | Number of procedures (non-lab) performed. |
| `num_medications` | Numeric | Number of distinct generic names administered during the encounter. |
| `number_outpatient` / `number_inpatient` / `number_emergency` | Numeric | Number of visits by the patient in the year preceding the encounter. |
| `max_glu_serum` / `A1Cresult` | Numeric (Ordinal) | Lab results mapped to integers (0 = None, 1 = Norm, 2/3 = Elevated bounds). |
| `medications` (e.g., `insulin`, `metformin`) | Numeric (Ordinal) | Administration changes for 21 drugs, mapped to integers (0=No, 1=Down, 2=Steady, 3=Up). |
| `change` / `diabetesMed` | Numeric | Binary flags for medication changes (1=Ch/Yes, 0=No). |
