"""
Steam Success Prediction - Model Optimization
================================================
Steps:
1. 5-Fold Cross Validation for stability
2. Hyperparameter Tuning (GridSearchCV)
3. Algorithm Comparison (Decision Tree vs Random Forest)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

INPUT_FILE = 'model_ready_dataset.csv'
REPORT_FILE = 'model_optimization_report.txt'
report_lines = []
def log(msg):
    print(msg)
    report_lines.append(msg)

log("=" * 70)
log("  STEAM SUCCESS PREDICTION - MODEL OPTIMIZATION")
log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)

df = pd.read_csv(INPUT_FILE)
df['target_3class'] = df['target_tier'].replace({3: 2})
review_metrics = ['pct_pos_total', 'num_reviews_total', 'peak_ccu']
all_features = [c for c in df.columns if c not in ['target_tier', 'target_3class']]
structural_features = [c for c in all_features if c not in review_metrics + ['release_year']]

# STEP 1: CROSS VALIDATION
def run_cv(X, y, name):
    log(f"\n  --- CV: {name} ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
    scores = cross_validate(rf, X, y, cv=cv, scoring=['accuracy', 'f1_macro'])
    log(f"  Macro-F1: {scores['test_f1_macro'].mean():.4f} +/- {scores['test_f1_macro'].std():.4f}")
    log(f"  Accuracy: {scores['test_accuracy'].mean():.4f} +/- {scores['test_accuracy'].std():.4f}")

run_cv(df[all_features], df['target_3class'], "3-Class + Reviews")
run_cv(df[structural_features], df['target_3class'], "3-Class Structural Only")

# STEP 2: TUNING
def run_tuning(X, y, name):
    log(f"\n  --- Tuning: {name} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [None, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), 
                        param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    log(f"  Best Params: {grid.best_params_}")
    log(f"  Best Macro-F1: {grid.best_score_:.4f}")

run_tuning(df[structural_features], df['target_3class'], "3-Class Structural Only")

# STEP 3: DT vs RF
X_train, X_test, y_train, y_test = train_test_split(df[structural_features], df['target_3class'], test_size=0.2, random_state=42, stratify=df['target_3class'])
dt = DecisionTreeClassifier(class_weight='balanced', random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1).fit(X_train, y_train)

log(f"\n  --- Comparison ---")
log(f"  DT Macro-F1: {dt.score(X_test, y_test):.4f}")
log(f"  RF Macro-F1: {rf.score(X_test, y_test):.4f}")

with open(REPORT_FILE, 'w', encoding='utf-8') as f: f.write('\n'.join(report_lines))
print(f"Report saved: {REPORT_FILE}")
