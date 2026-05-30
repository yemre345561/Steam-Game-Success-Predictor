import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import os

# Create reporting function
def log(msg):
    print(msg)
    with open("model_training_report.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear old report
if os.path.exists("model_training_report.txt"):
    os.remove("model_training_report.txt")

log("="*70)
log("  STEAM SUCCESS PREDICTION - RANDOM FOREST TRAINING")
log(f"  Started: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("="*70)

# Load data
df = pd.read_csv('model_ready_dataset.csv')
target_col = 'target_tier'

# Experiment with 3-class consolidation (Merging Mega Hit into Successful)
df['target_3class'] = df[target_col].replace({3: 2})

# Define features
review_metrics = ['pct_pos_total', 'num_reviews_total', 'peak_ccu']
temporal_leak = ['release_year']  # Excluded: temporal metadata, not a developer-controllable feature
all_features = [c for c in df.columns if c not in [target_col, 'target_3class']]
structural_features = [c for c in all_features if c not in review_metrics + temporal_leak]

def run_experiment(exp_name, X, y, class_labels):
    log(f"\n{'='*70}\n  {exp_name}\n{'='*70}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    log(f"\n  Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    log("\n  Classification Report:")
    log(classification_report(y_test, y_pred, target_names=class_labels))
    
    # Confusion Matrix formatting
    cm = confusion_matrix(y_test, y_pred)
    log("\n  Confusion Matrix:")
    log(f"  Predicted -> {'':>14} " + " ".join([f"{l:>14}" for l in class_labels]))
    log("  " + "-"*80)
    for i, label in enumerate(class_labels):
        row = " ".join([f"{val:>14,}" for val in cm[i]])
        log(f"  {label:<16} {row}")

    # Top Features
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    log("\n  Top 20 Features:")
    for i, (name, val) in enumerate(importances[:20].items()):
        log(f"  {i+1:<2} {name:<30} {val:.4f}")

# --- EXPERIMENTS ---
labels_4 = ['Flop', 'Niche Hit', 'Commercial Hit', 'Mega Hit']
labels_3 = ['Flop', 'Niche Hit', 'Successful']

run_experiment("EXP 1: 4-Class + Reviews", df[all_features], df[target_col], labels_4)
run_experiment("EXP 2: 4-Class Structural", df[structural_features], df[target_col], labels_4)
run_experiment("EXP 3: 3-Class + Reviews", df[all_features], df['target_3class'], labels_3)
run_experiment("EXP 4: 3-Class Structural", df[structural_features], df['target_3class'], labels_3)
