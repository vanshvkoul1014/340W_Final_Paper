"""
=============================================================================
Credit Risk Prediction with Dual Explainable AI (SHAP + LIME)
Research Paper Code — DS 340W — Vansh Vinod Koul
Penn State University, Spring 2026

Extension of: Chang et al. (2024), Risks, 12(11), 174

Pipeline:
  1. Load & merge data
  2. Target variable engineering
  3. Preprocessing (outlier removal, encoding, scaling)
  4. 70 / 10 / 20  train / validation / test split + SMOTE
  5. Train 6 models (LR, RF, AdaBoost, XGBoost, LightGBM, Neural Network)
  6. Evaluate on validation and test sets
  7. SHAP analysis (TreeSHAP + KernelSHAP)
  8. LIME analysis (instance-level + global average)
  9. SHAP vs LIME comparison
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import shap
import lime
import lime.lime_tabular

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 65)
print("STEP 1: Loading Data")
print("=" * 65)

app    = pd.read_csv('application_record.csv')
credit = pd.read_csv('credit_record.csv')

print(f"application_record : {app.shape[0]:,} rows, {app.shape[1]} columns")
print(f"credit_record      : {credit.shape[0]:,} rows, {credit.shape[1]} columns")

# ============================================================
# STEP 2: TARGET VARIABLE ENGINEERING
# ============================================================
print("\n" + "=" * 65)
print("STEP 2: Engineering Target Variable")
print("=" * 65)

# STATUS codes: X/C/0 → good months; 1-5 → increasing delinquency
# Target = 1 if customer ever reached 60+ days overdue (status ≥ 2)
status_map = {'X': 0, 'C': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
credit['STATUS_NUM'] = credit['STATUS'].map(status_map)

credit_agg = credit.groupby('ID').agg(
    max_status  = ('STATUS_NUM', 'max'),
    num_months  = ('MONTHS_BALANCE', 'count'),
    oldest_month= ('MONTHS_BALANCE', 'min')
).reset_index()

credit_agg['TARGET'] = (credit_agg['max_status'] >= 2).astype(int)

print(f"Unique customers : {credit_agg.shape[0]:,}")
print(f"Good (0)         : {(credit_agg['TARGET']==0).sum():,}")
print(f"Bad  (1)         : {(credit_agg['TARGET']==1).sum():,}")
print(f"Default rate     : {credit_agg['TARGET'].mean()*100:.2f}%")

# ============================================================
# STEP 3: MERGE
# ============================================================
print("\n" + "=" * 65)
print("STEP 3: Merging Datasets")
print("=" * 65)

df = app.merge(credit_agg[['ID', 'TARGET']], on='ID', how='inner')
df = df.drop_duplicates(subset='ID', keep='first')
print(f"Merged dataset : {df.shape[0]:,} rows, default rate {df['TARGET'].mean()*100:.2f}%")

# ============================================================
# STEP 4: PREPROCESSING
# ============================================================
print("\n" + "=" * 65)
print("STEP 4: Preprocessing")
print("=" * 65)

df = df.drop('ID', axis=1)

# Age and employment in years
df['AGE']            = (-df['DAYS_BIRTH']    / 365.25).astype(int)
df['EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'].apply(lambda x: 0 if x > 0 else -x / 365.25)
df = df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1)

# Drop useless/high-missing columns
if 'FLAG_MOBIL' in df.columns:
    df = df.drop('FLAG_MOBIL', axis=1)
if 'OCCUPATION_TYPE' in df.columns and df['OCCUPATION_TYPE'].isnull().mean() > 0.3:
    df = df.drop('OCCUPATION_TYPE', axis=1)

# Impute remaining missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = (df[col].fillna(df[col].mode()[0])
                   if df[col].dtype == 'object'
                   else df[col].fillna(df[col].median()))

# IQR outlier removal on skewed numeric columns
print(f"Before outlier removal : {df.shape[0]:,} rows")
for col in ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS']:
    if col in df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df  = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
print(f"After  outlier removal : {df.shape[0]:,} rows")

# One-hot encode categoricals
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
df       = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"Final dataset          : {df.shape[0]:,} rows, {df.shape[1]-1} features")

# ============================================================
# STEP 5: 70 / 10 / 20  SPLIT  +  SMOTE
# ============================================================
print("\n" + "=" * 65)
print("STEP 5: 70 / 10 / 20 Split  +  SMOTE")
print("=" * 65)

X = df.drop('TARGET', axis=1)
y = df['TARGET']

# 70 % train  |  30 % temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

# temp → 10 % val  |  20 % test   (i.e. 1/3 val, 2/3 test of the 30 %)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

print(f"Train : {X_train.shape[0]:,}  ({X_train.shape[0]/len(X)*100:.0f}%)"
      f"  |  bad rate {y_train.mean()*100:.2f}%")
print(f"Val   : {X_val.shape[0]:,}   ({X_val.shape[0]/len(X)*100:.0f}%)"
      f"  |  bad rate {y_val.mean()*100:.2f}%")
print(f"Test  : {X_test.shape[0]:,}  ({X_test.shape[0]/len(X)*100:.0f}%)"
      f"  |  bad rate {y_test.mean()*100:.2f}%")

# SMOTE on training only
smote            = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE → train : {X_train_sm.shape[0]:,}"
      f"  |  bad rate {y_train_sm.mean()*100:.1f}%")

# Scale (for LR + NN only; trees get unscaled)
scaler         = StandardScaler()
X_train_sc     = scaler.fit_transform(X_train_sm)
X_val_sc       = scaler.transform(X_val)
X_test_sc      = scaler.transform(X_test)

feature_names  = X.columns.tolist()

# ============================================================
# STEP 6: TRAIN ALL 6 MODELS
# ============================================================
print("\n" + "=" * 65)
print("STEP 6: Training All 6 Models")
print("=" * 65)

models_cfg = {
    'Logistic Regression': {
        'model' : LogisticRegression(max_iter=1000, random_state=42),
        'scaled': True },
    'Random Forest': {
        'model' : RandomForestClassifier(n_estimators=200, max_depth=15,
                                         random_state=42, n_jobs=-1),
        'scaled': False },
    'AdaBoost': {
        'model' : AdaBoostClassifier(n_estimators=100, random_state=42),
        'scaled': False },
    'XGBoost': {
        'model' : xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                     learning_rate=0.1, use_label_encoder=False,
                                     eval_metric='logloss', random_state=42,
                                     n_jobs=-1),
        'scaled': False },
    'LightGBM': {
        'model' : lgb.LGBMClassifier(n_estimators=200, max_depth=6,
                                      learning_rate=0.1, random_state=42,
                                      n_jobs=-1, verbose=-1),
        'scaled': False },
    'Neural Network': {
        'model' : MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                activation='relu', max_iter=300,
                                early_stopping=True, validation_fraction=0.1,
                                random_state=42),
        'scaled': True },
}

def get_metrics(y_true, y_pred, y_prob):
    return dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, zero_division=0),
        recall    = recall_score(y_true, y_pred),
        f1        = f1_score(y_true, y_pred),
        roc_auc   = roc_auc_score(y_true, y_prob),
        mcc       = matthews_corrcoef(y_true, y_pred),
    )

results       = []
trained       = {}   # stores model + test predictions

for name, cfg in models_cfg.items():
    print(f"\n  Training: {name}")
    m   = cfg['model']
    sc  = cfg['scaled']

    Xtr = X_train_sc if sc else X_train_sm
    Xv  = X_val_sc   if sc else X_val
    Xte = X_test_sc  if sc else X_test

    m.fit(Xtr, y_train_sm)

    val_pred  = m.predict(Xv);       val_prob  = m.predict_proba(Xv)[:,1]
    test_pred = m.predict(Xte);      test_prob = m.predict_proba(Xte)[:,1]

    vm = get_metrics(y_val,  val_pred,  val_prob)
    tm = get_metrics(y_test, test_pred, test_prob)

    results.append({
        'Model'         : name,
        'Val MCC'       : round(vm['mcc'],       4),
        'Test Accuracy' : round(tm['accuracy'],  4),
        'Test Precision': round(tm['precision'], 4),
        'Test Recall'   : round(tm['recall'],    4),
        'Test F1'       : round(tm['f1'],        4),
        'Test ROC-AUC'  : round(tm['roc_auc'],   4),
        'Test MCC'      : round(tm['mcc'],       4),
    })

    trained[name] = {'model': m, 'scaled': sc,
                     'y_pred': test_pred, 'y_prob': test_prob}

    cm = confusion_matrix(y_test, test_pred)
    print(f"    Val  MCC  : {vm['mcc']:.4f}")
    print(f"    Test MCC  : {tm['mcc']:.4f}  |  Acc {tm['accuracy']:.4f}"
          f"  |  Recall {tm['recall']:.4f}")
    print(f"    CM → TN={cm[0,0]:,}  FP={cm[0,1]:,}  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# ============================================================
# STEP 7: RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("STEP 7: Results Summary  (sorted by Test MCC)")
print("=" * 65)

results_df = pd.DataFrame(results).sort_values('Test MCC', ascending=False)
print("\n" + results_df.to_string(index=False))
results_df.to_csv('model_results.csv', index=False)

best = results_df.iloc[0]
print(f"\n  ★  Best model : {best['Model']}"
      f"  |  Test MCC {best['Test MCC']:.4f}"
      f"  |  Test Acc {best['Test Accuracy']:.4f}")

# Save for external use
joblib.dump(trained,   'trained_models.pkl')
joblib.dump(scaler,    'scaler.pkl')
joblib.dump((X_test, X_test_sc, y_test), 'test_data.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

# ============================================================
# STEP 8: SHAP ANALYSIS
# ============================================================
print("\n" + "=" * 65)
print("STEP 8: SHAP Analysis")
print("=" * 65)

shap_results = {}

# ---- tree-based models (exact TreeSHAP) ----
for name in ['XGBoost', 'LightGBM', 'Random Forest']:
    print(f"\n  SHAP → {name}")
    m   = trained[name]['model']
    exp = shap.TreeExplainer(m)
    sv  = exp.shap_values(X_test)

    # handle 3-D output (RF returns shape [n, features, 2])
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]

    imp = pd.DataFrame({'Feature': feature_names,
                        'Mean_SHAP': np.abs(sv).mean(axis=0)}) \
            .sort_values('Mean_SHAP', ascending=False)
    shap_results[name] = imp
    print(f"    Top 5 : {imp['Feature'].head(5).tolist()}")

# ---- Neural Network (KernelSHAP approximation) ----
print(f"\n  SHAP → Neural Network  (KernelSHAP, 50 instances)")
nn_m   = trained['Neural Network']['model']
bg     = X_test_sc[:30]
k_exp  = shap.KernelExplainer(nn_m.predict_proba, bg)
sv_nn  = k_exp.shap_values(X_test_sc[:50], nsamples=50)
if isinstance(sv_nn, list):
    sv_nn = sv_nn[1]
elif isinstance(sv_nn, np.ndarray) and sv_nn.ndim == 3:
    sv_nn = sv_nn[:, :, 1]

imp_nn = pd.DataFrame({'Feature': feature_names,
                       'Mean_SHAP': np.abs(sv_nn).mean(axis=0)}) \
           .sort_values('Mean_SHAP', ascending=False)
shap_results['Neural Network'] = imp_nn
print(f"    Top 5 : {imp_nn['Feature'].head(5).tolist()}")

# Save SHAP tables
for name, df_shap in shap_results.items():
    fname = f"shap_{name.lower().replace(' ', '_')}.csv"
    df_shap.to_csv(fname, index=False)
    print(f"  Saved {fname}")

# ============================================================
# STEP 9: LIME ANALYSIS
# ============================================================
print("\n" + "=" * 65)
print("STEP 9: LIME Analysis")
print("=" * 65)

lime_exp = lime.lime_tabular.LimeTabularExplainer(
    training_data = X_test_sc,
    feature_names = feature_names,
    class_names   = ['Good', 'Bad'],
    mode          = 'classification',
    random_state  = 42
)

default_idx = np.where(y_test.values == 1)[0]
good_idx    = np.where(y_test.values == 0)[0]

lime_results = {}

for name in ['XGBoost', 'LightGBM', 'Neural Network']:
    m  = trained[name]['model']
    sc = trained[name]['scaled']

    def predict_fn(X_arr):
        return m.predict_proba(X_arr if sc else scaler.inverse_transform(X_arr))

    print(f"\n  LIME → {name}")

    # Instance-level: one defaulter, one good customer
    for label, idx_arr in [('DEFAULT', default_idx), ('GOOD', good_idx)]:
        idx = idx_arr[0]
        e   = lime_exp.explain_instance(X_test_sc[idx], predict_fn, num_features=10)
        top = e.as_list()[:5]
        print(f"    {label} customer top features:")
        for feat, w in top:
            print(f"      {'↑' if w > 0 else '↓'} {feat[:40]:<40}  w={w:+.4f}")

    # Global LIME: average over 50 random test instances
    sample_idx  = np.random.RandomState(42).choice(len(X_test_sc), 50, replace=False)
    lime_global = {}
    for idx in sample_idx:
        e = lime_exp.explain_instance(X_test_sc[idx], predict_fn, num_features=10)
        for feat, w in e.as_list():
            key = feat.split(' ')[0].strip('<>=!')
            if key in feature_names:
                lime_global.setdefault(key, []).append(abs(w))

    lime_top = sorted(lime_global.items(),
                      key=lambda x: np.mean(x[1]), reverse=True)
    lime_results[name] = lime_top
    print(f"    Global LIME top 5 : {[f[0] for f in lime_top[:5]]}")

# ============================================================
# STEP 10: SHAP vs LIME COMPARISON
# ============================================================
print("\n" + "=" * 65)
print("STEP 10: SHAP vs LIME Comparison")
print("=" * 65)

for name in ['XGBoost', 'LightGBM', 'Neural Network']:
    shap_top5 = set(shap_results[name]['Feature'].head(5))
    lime_top5 = set(f[0] for f in lime_results[name][:5])
    overlap   = shap_top5 & lime_top5

    print(f"\n  {name}")
    print(f"    SHAP top 5 : {sorted(shap_top5)}")
    print(f"    LIME top 5 : {sorted(lime_top5)}")
    print(f"    Overlap    : {len(overlap)}/5  →  {overlap if overlap else 'None'}")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 65)
print("COMPLETE")
print("=" * 65)
print("""
Output files
  model_results.csv              — all 6 model metrics (val + test)
  shap_xgboost.csv               — XGBoost SHAP feature importances
  shap_lightgbm.csv              — LightGBM SHAP feature importances
  shap_random_forest.csv         — Random Forest SHAP feature importances
  shap_neural_network.csv        — Neural Network SHAP feature importances
  trained_models.pkl             — all 6 trained model objects
  scaler.pkl                     — fitted StandardScaler
  test_data.pkl                  — (X_test, X_test_sc, y_test)
  feature_names.pkl              — list of feature names
""")
