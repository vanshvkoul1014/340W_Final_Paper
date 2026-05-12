# Credit Risk Prediction with Dual Explainable AI (SHAP & LIME)

**DS 340W Final Paper Project**  
**Author:** Vansh Vinod Koul  
**Institution:** College of Engineering, The Pennsylvania State University  

This repository contains the code and supporting materials for a credit card default prediction project that replicates and extends the machine learning pipeline from Chang et al. (2024). The project compares six classification models on the Kaggle Credit Card Approval Prediction dataset and adds a dual explainability layer using **SHAP** and **LIME**.

The main goal is not only to compare predictive performance, but also to evaluate whether SHAP and LIME tell the same explanation story for credit-risk models. The results show that the two methods often identify different top features, suggesting that they should be treated as complementary rather than interchangeable.

---

## Repository Contents

```text
340W_Final_Paper/
├── 340W_Research_Paper_Final.pdf      # Final research paper
├── credit_risk_replication.py         # Full reproducible ML + XAI pipeline
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Ignored files and outputs
```

After running the script, the following output files are generated:

```text
model_results.csv
shap_xgboost.csv
shap_lightgbm.csv
shap_random_forest.csv
shap_neural_network.csv
trained_models.pkl
scaler.pkl
test_data.pkl
feature_names.pkl
```

---

## Project Summary

Credit card default prediction is a highly imbalanced classification problem. In this dataset, the positive default class represents only about **1.69%** of the cleaned modeling sample. Because raw accuracy can be misleading in such a skewed dataset, the project uses **Matthews Correlation Coefficient (MCC)** as the primary evaluation metric.

The pipeline performs the following steps:

1. Load the Kaggle credit application and credit history files.
2. Engineer a binary default target based on delinquency status.
3. Merge applicant-level and credit-status-level records.
4. Clean, encode, and preprocess the dataset.
5. Split the data into **70% train, 10% validation, and 20% test**.
6. Apply **SMOTE only to the training fold**.
7. Train six classification models.
8. Evaluate all models on the untouched test fold.
9. Run SHAP analysis for model-level feature importance.
10. Run LIME analysis for local explanation behavior.
11. Compare SHAP and LIME top-feature overlap.

---

## Dataset

This project uses the public Kaggle **Credit Card Approval Prediction** dataset.

The two required files are:

```text
application_record.csv
credit_record.csv
```

Place both CSV files in the project root directory before running the script.

Expected raw input sizes:

| File | Rows | Columns |
|---|---:|---:|
| `application_record.csv` | 438,557 | 18 |
| `credit_record.csv` | 1,048,575 | 3 |

After preprocessing and outlier removal, the final modeling table contains:

| Item | Value |
|---|---:|
| Final records | 34,455 |
| Features | 28 |
| Default rate | 1.69% |

---

## Models Compared

The project trains and evaluates six models:

1. Logistic Regression
2. Random Forest
3. AdaBoost
4. XGBoost
5. LightGBM
6. Neural Network / Multilayer Perceptron

The neural network uses a three-layer architecture with hidden layers of size **128, 64, and 32**, along with regularization and early stopping.

---

## Final Results

Test-fold results sorted by **Test MCC**:

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Test MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Neural Network** | **97.04%** | **0.240** | **0.357** | **0.287** | **0.702** | **0.278** |
| LightGBM | 97.56% | 0.281 | 0.296 | 0.288 | 0.661 | 0.276 |
| XGBoost | 97.53% | 0.257 | 0.252 | 0.254 | 0.653 | 0.242 |
| Random Forest | 95.79% | 0.134 | 0.278 | 0.181 | 0.697 | 0.173 |
| Logistic Regression | 90.71% | 0.040 | 0.200 | 0.067 | 0.543 | 0.055 |
| AdaBoost | 87.30% | 0.024 | 0.165 | 0.042 | 0.531 | 0.020 |

**Best model:** Neural Network  
**Test MCC:** 0.278  
**Test Accuracy:** 97.04%

The overall result is consistent with the paper's main finding: the enhanced Neural Network performs best by MCC, while LightGBM and XGBoost remain close competitors.

---

## SHAP Results

The script computes SHAP feature importances for XGBoost, LightGBM, Random Forest, and Neural Network.

Top SHAP features:

| Model | Top 5 SHAP Features |
|---|---|
| XGBoost | Income Type: Pensioner, Education: Secondary, Education: Higher, Family Members, Total Income |
| LightGBM | Family Members, Education: Higher, Education: Secondary, Income Type: Pensioner, Single / Not Married |
| Random Forest | Education: Secondary, Education: Higher, Phone Flag, Income Type: Working, Income Type: Pensioner |
| Neural Network | Education: Secondary, Education: Higher, Single / Not Married, Family Members, Children Count |

---

## LIME Results

LIME is used to generate local explanations and then averaged across 50 sampled test instances to provide a global comparison against SHAP.

Top global LIME features:

| Model | Top 5 Global LIME Features |
|---|---|
| XGBoost | Income Type: Pensioner, Family Members, Email Flag, Married Status, Income Type: Student |
| LightGBM | Widow Status, Separated Status, Email Flag, Family Members, Children Count |
| Neural Network | Family Members, Income Type: Student, Separated Status, Widow Status, Email Flag |

---

## SHAP vs LIME Comparison

| Model | SHAP/LIME Top-5 Overlap |
|---|---:|
| XGBoost | 2 / 5 |
| LightGBM | 1 / 5 |
| Neural Network | 1 / 5 |

This supports the main argument of the paper: **SHAP and LIME are not interchangeable**. SHAP is more useful for population-level model behavior, while LIME is more useful for instance-level explanation.

---

## Installation

### Recommended Setup: Virtual Environment

Using a virtual environment is recommended, especially on macOS.

```bash
cd 340W_Final_Paper
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then run:

```bash
python3 credit_risk_replication.py
```

To exit the virtual environment:

```bash
deactivate
```

---

## macOS Setup Notes

On newer macOS/Homebrew Python installations, you may see this error when trying to install packages globally:

```text
error: externally-managed-environment
```

This happens because Homebrew protects the system-managed Python environment. The recommended fix is to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

If XGBoost fails with an error involving `libomp.dylib`, install the OpenMP runtime:

```bash
brew install libomp
```

Then run the script again:

```bash
python3 credit_risk_replication.py
```

If the issue persists, run these commands in the same terminal session:

```bash
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
python3 credit_risk_replication.py
```

---

## Running Without a Virtual Environment

This is not recommended on macOS/Homebrew Python because it can interfere with system-managed Python packages. However, if needed:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
python3 credit_risk_replication.py
```

The virtual environment method is the safer and cleaner option.

---

## Reproducing Results

To reproduce the results reported in this project:

1. Use Python 3.8 or higher.
2. Install the package versions listed in `requirements.txt`.
3. Place the original Kaggle CSV files in the project root:
   - `application_record.csv`
   - `credit_record.csv`
4. Run the script without modifying the code:

```bash
python3 credit_risk_replication.py
```

The script fixes random seeds using `random_state=42` where supported. However, exact metric values may vary slightly across Python versions, operating systems, and package versions, especially for SMOTE, neural network training, SHAP, and LIME.

The expected reproduced pattern is:

- Neural Network ranks highest by MCC.
- LightGBM and XGBoost remain close behind.
- SHAP and LIME show limited overlap in their top-ranked features.
- MCC provides a more meaningful evaluation than raw accuracy on this imbalanced dataset.

---

## Why MCC Is Used

The default class is extremely rare, so accuracy alone can be misleading. A model could achieve very high accuracy by predicting nearly every applicant as non-default. MCC is more appropriate because it accounts for true positives, true negatives, false positives, and false negatives in a single score.

This is especially important for credit-risk modeling because identifying the minority default class is the main task.

---

## Main Conclusion

This project shows that a properly regularized neural network can compete with and slightly outperform tree-based ensemble models on this credit-risk dataset when evaluated using MCC. It also shows that SHAP and LIME often produce different top-feature explanations, meaning they should be used together rather than treated as substitutes.

SHAP is better suited for understanding global model behavior, while LIME is better suited for explaining individual predictions.

---

## Reference Paper

The full research paper is included in this repository:

```text
340W_Research_Paper_Final.pdf
```

Title:

**A Side-by-Side Look at SHAP and LIME: Bringing Explainability to Six Models for Credit Card Default Risk**

---

## Notes

- The Kaggle CSV files are not included in this repository. Download them from: https://www.kaggle.com/code/rikdifos/eda-vintage-analysis/input
- Generated output files are ignored by Git unless intentionally added.
- Runtime may be longer during SHAP analysis, especially for Random Forest and Neural Network explanations.
- The first Matplotlib run may display: `Matplotlib is building the font cache; this may take a moment.` This is normal.
