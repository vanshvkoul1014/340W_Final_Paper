# Credit Risk Prediction with Dual Explainable AI (SHAP & LIME)

**DS 340W — Penn State University — Spring 2026**
**Author:** Vansh Vinod Koul

---

## Overview

This project replicates and extends Chang et al. (2024) by applying dual Explainable AI
(SHAP + LIME) across six ML/DL models on the Kaggle Credit Card Dataset. The goal is to
show that SHAP and LIME answer different but complementary questions, and that both should
be used together in production credit scoring systems.

**Three Research Contributions:**
1. **Dual XAI** — First study comparing SHAP and LIME side-by-side across all six models on this dataset. XGBoost shows 2/5 overlap, LightGBM and Neural Network show 1/5.
2. **Improved Neural Network** — 3-layer MLP with dropout and early stopping achieves 97.04% accuracy and MCC 0.278, vs the parent paper's 87.2%.
3. **Correct SMOTE Placement** — SMOTE applied to training fold only after the 70/10/20 split, avoiding data leakage that inflated the parent paper's MCC to 0.986.

---

## Results Summary

| Model | Test Accuracy | Precision | Recall | F1 | ROC-AUC | MCC |
|---|---|---|---|---|---|---|
| **Neural Network** | **97.04%** | **0.240** | **0.357** | **0.287** | **0.702** | **0.278** |
| LightGBM | 97.56% | 0.281 | 0.296 | 0.288 | 0.661 | 0.276 |
| XGBoost | 97.53% | 0.257 | 0.252 | 0.254 | 0.653 | 0.242 |
| Random Forest | 95.79% | 0.134 | 0.278 | 0.181 | 0.697 | 0.173 |
| Logistic Regression | 90.71% | 0.040 | 0.200 | 0.067 | 0.543 | 0.055 |
| AdaBoost | 87.30% | 0.024 | 0.165 | 0.042 | 0.531 | 0.020 |

> MCC (Matthews Correlation Coefficient) is the primary metric. It is the fairest single metric
> for heavily imbalanced datasets — a model that predicts the majority class for every instance
> scores MCC = 0 regardless of accuracy.

---

## Dataset

The dataset is publicly available on Kaggle. You need two CSV files:

1. Go to: https://www.kaggle.com/code/rikdifos/eda-vintage-analysis/input
2. Download `application_record.csv` and `credit_record.csv`
3. Place both files in the same folder as `credit_risk_replication.py`

**Dataset summary:**
- `application_record.csv` — 438,557 rows, demographic and financial features per applicant
- `credit_record.csv` — monthly credit status codes per customer
- After cleaning: 34,455 customers, 28 features, 1.69% default rate

---

## Requirements

### Python Version
Python 3.8 or higher is required. To check your version:
```bash
python --version
```

### Dependencies
All required packages are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```

If you get permission errors, try:
```bash
pip install -r requirements.txt --user
```

If you are using a virtual environment (recommended):
```bash
# Create virtual environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

Once set up, your folder should look like this:

```
credit-risk-xai/
├── application_record.csv          ← download from Kaggle
├── credit_record.csv               ← download from Kaggle
├── credit_risk_replication.py      ← main script
├── requirements.txt                ← dependencies
└── README.md                       ← this file
```

After running the script, the following output files will be created automatically:

```
credit-risk-xai/
├── model_results.csv               ← metrics for all 6 models
├── shap_xgboost.csv                ← XGBoost SHAP feature importances
├── shap_lightgbm.csv               ← LightGBM SHAP feature importances
├── shap_random_forest.csv          ← Random Forest SHAP feature importances
├── shap_neural_network.csv         ← Neural Network SHAP feature importances
├── trained_models.pkl              ← saved model objects (not pushed to GitHub)
├── scaler.pkl                      ← fitted StandardScaler (not pushed to GitHub)
├── test_data.pkl                   ← test split arrays (not pushed to GitHub)
└── feature_names.pkl               ← list of feature names (not pushed to GitHub)
```

---

## How to Run

### Step 1 — Open a terminal

- **Mac/Linux:** Open Terminal
- **Windows:** Open Command Prompt or PowerShell
- **VS Code:** Open the integrated terminal with `Ctrl + backtick`

### Step 2 — Navigate to the project folder

```bash
cd path/to/credit-risk-xai
```

For example:
```bash
cd Documents/credit-risk-xai
```

### Step 3 — Run the script

```bash
python credit_risk_replication.py
```

If `python` does not work, try:
```bash
python3 credit_risk_replication.py
```

### Step 4 — Watch the output

The script prints progress for each step as it runs. A full run takes approximately
5–15 minutes depending on your machine. You will see output like:

```
=================================================================
STEP 1: Loading Data
=================================================================
application_record : 438,557 rows, 18 columns
credit_record      : 1,048,575 rows, 3 columns

=================================================================
STEP 2: Engineering Target Variable
...

=================================================================
STEP 6: Training All 6 Models
=================================================================
  Training Logistic Regression... MCC=0.0554
  Training Random Forest... MCC=0.1734
  Training AdaBoost... MCC=0.0201
  Training XGBoost... MCC=0.2419
  Training LightGBM... MCC=0.2758
  Training Neural Network... MCC=0.2778

=================================================================
STEP 7: Results Summary  (sorted by Test MCC)
...

STEP 8: SHAP Analysis
  SHAP -> XGBoost...
  SHAP -> LightGBM...
  SHAP -> Random Forest...
  SHAP -> Neural Network (KernelSHAP)...

STEP 9: LIME Analysis
  LIME [XGBoost] top5: [...]
  LIME [LightGBM] top5: [...]
  LIME [Neural Network] top5: [...]

STEP 10: SHAP vs LIME Comparison
  XGBoost: overlap 2/5
  LightGBM: overlap 1/5
  Neural Network: overlap 1/5
```

---

## Pipeline Description

The script runs ten steps end to end:

| Step | Description |
|---|---|
| 1 | Load both CSV files |
| 2 | Engineer target variable — customers 60+ days overdue labeled as bad (TARGET=1) |
| 3 | Merge application and credit records on customer ID |
| 4 | Preprocess — convert age/employment to years, drop constant columns, impute missing values, IQR outlier removal, one-hot encoding |
| 5 | Split 70% train / 10% validation / 20% test (stratified), apply SMOTE to training only |
| 6 | Train all 6 models with standardization for LR and NN, raw features for tree models |
| 7 | Evaluate — Accuracy, Precision, Recall, F1, ROC-AUC, MCC on both validation and test sets |
| 8 | SHAP — TreeSHAP for XGBoost/LightGBM/Random Forest, KernelSHAP for Neural Network |
| 9 | LIME — global average feature importance across 50 sampled test instances per model |
| 10 | SHAP vs LIME comparison — top-5 feature overlap per model |

---

## Troubleshooting

**ModuleNotFoundError** — a package is missing. Run:
```bash
pip install -r requirements.txt
```

**MemoryError** — your machine may not have enough RAM for the full dataset.
The script uses approximately 4GB of RAM at peak (during SMOTE and model training).
Close other applications and try again.

**KernelSHAP is slow** — KernelSHAP for the Neural Network can take 1–2 minutes.
This is expected. The script uses a small background sample (30 instances) and
evaluates 50 instances to keep runtime manageable.

**FileNotFoundError: application_record.csv** — the CSV files are not in the same
folder as the script. Make sure both CSV files are in the project root directory.

---

## Reproducing Exact Results

To reproduce the exact numbers reported in the paper:
- Use Python 3.8+
- Install the exact package versions in `requirements.txt`
- Place the original Kaggle CSV files in the project root
- Run `python credit_risk_replication.py` without any modifications
- All random seeds are fixed to `random_state=42`

---

## References

1. V. Chang et al. (2024). *Credit Risk Prediction Using ML and DL*. Risks, 12(11), 174.
2. P. E. de Lange et al. (2022). *Explainable AI for Credit Assessment in Banks*. JRFM, 15(12), 556.
3. F. M. Talaat et al. (2024). *Toward Interpretable Credit Scoring*. Neural Comput. Appl., 36, 4847–4865.
4. S. M. Lundberg and S.-I. Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
5. M. T. Ribeiro et al. (2016). *'Why Should I Trust You?'*. ACM SIGKDD.
6. N. V. Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 16, 321–357.
