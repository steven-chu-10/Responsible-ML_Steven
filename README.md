# COMPAS Recidivism Analysis — Python Implementation

A Python replication of ProPublica's COMPAS recidivism risk score analysis, originally performed in R. This notebook reproduces the full analytical workflow: data loading, preprocessing, exploratory data analysis, logistic regression modeling, and fairness diagnostics across racial groups.

The analysis is part of Individual Homework 1 for DNSC 6330: Responsible Machine Learning at The George Washington University.

---

## Purpose

The COMPAS algorithm assigns recidivism risk scores to criminal defendants. ProPublica's investigation found that the algorithm produced significantly higher false positive rates for Black defendants than for white defendants, even after controlling for other factors.

This notebook replicates that analysis in Python, reproducing each step of the original R workflow to verify the same findings and examine where the model's predictions diverge across racial groups.

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, filtering, and manipulation |
| `numpy` | Numerical operations and array handling |
| `statsmodels` | Logistic regression (GLM with binomial family) |
| `matplotlib` | Bar charts and visualizations |
| `scikit-learn` | Model pipeline, preprocessing, gradient-boosted tree, evaluation metrics |
| `lime` | Local interpretable model-agnostic explanations |
| `warnings` | Suppressing non-critical warnings |

---

## How to Reproduce

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies

```bash
pip install pandas numpy statsmodels matplotlib scikit-learn lime
```

### 3. Run the notebook

Open `Lecture_01_Alignment_Python.ipynb` in Jupyter or Google Colab and run all cells top to bottom. The dataset is loaded directly from ProPublica's public GitHub repository, so no local data files are needed.

```
Dataset URL: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
```

---

## Analytical Pipeline

### 1. Data Loading and Preprocessing
- Loads the raw COMPAS dataset (7,214 rows) directly from GitHub
- Selects relevant columns: demographics, charge details, COMPAS scores, recidivism outcome
- Filters to cases where screening occurred within 30 days of arrest, excludes invalid recidivism flags, ordinary traffic charges, and missing scores
- Converts datetime columns and constructs categorical variables with explicit reference levels matching the original R workflow

### 2. Exploratory Data Analysis
- Computes the correlation between jail length of stay and COMPAS decile score
- Plots decile score distributions separately for Black and white defendants to visualize score disparities

### 3. Logistic Regression Model
- Fits a binomial GLM predicting whether a defendant receives a Medium or High COMPAS score
- Predictors: gender, age category, race, prior arrests, charge degree, two-year recidivism outcome
- Reference levels: Male, 25-45 age group, Caucasian — matching the R model specification
- Computes the race effect by converting the African-American coefficient from log-odds to a predicted probability at baseline

### 4. Model Diagnostics and Fairness Evaluation
- Computes confusion matrix counts (TP, TN, FP, FN) separately for each racial group
- Derives accuracy, precision, recall, false positive rate (FPR), and false negative rate (FNR) per group
- Reproduces the core ProPublica finding: Black defendants face a substantially higher FPR than white defendants

---

## Key Findings

The logistic regression and fairness diagnostics reproduce the central result from ProPublica's investigation. Black defendants in the COMPAS dataset face a false positive rate of approximately **0.367**, meaning they are flagged as high-risk but do not reoffend at nearly twice the rate of white defendants (FPR ≈ 0.104). This disparity persists even controlling for charge degree, prior arrests, and recidivism outcome, which points to proxy discrimination rather than direct use of race.

