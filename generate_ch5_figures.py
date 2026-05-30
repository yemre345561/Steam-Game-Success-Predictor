"""
Chapter 5: ML Results - Thesis-Grade Figures
"""
import pandas as pd, numpy as np, matplotlib, os, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, classification_report, f1_score,
                             accuracy_score, precision_score, recall_score)
warnings.filterwarnings('ignore')

OUT = 'chapter5_ml_results_figures'
os.makedirs(OUT, exist_ok=True)

# --- STYLE ---
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11, 'figure.dpi': 300,
    'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'figure.facecolor': '#FFFFFF', 'axes.facecolor': '#FAFBFC',
    'axes.edgecolor': '#DEE2E6', 'grid.color': '#E9ECEF', 'grid.alpha': 0.7,
})
PAL = {'flop': '#E74C3C', 'niche': '#3498DB', 'success': '#2ECC71',
       'dark': '#1B2A4A', 'mid': '#2E5090', 'accent': '#E8A838',
       'light': '#7AB8F5', 'bg': '#F8F9FA', 'dt': '#E8A838', 'rf': '#2E5090'}
CLASS_COLORS = [PAL['flop'], PAL['niche'], PAL['success']]
LABELS = ['Flop', 'Niche Hit', 'Successful']

# --- DATA & MODEL ---
print("="*60)
print("  CHAPTER 5 FIGURE GENERATOR (Thesis-Grade)")
print("="*60)
df = pd.read_csv('model_ready_dataset.csv')
df['target_3class'] = df['target_tier'].replace({3: 2})
review_metrics = ['pct_pos_total', 'num_reviews_total', 'peak_ccu']
structural = [c for c in df.columns if c not in ['target_tier','target_3class'] + review_metrics + ['release_year']]
X, y = df[structural], df['target_3class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"  Features: {len(structural)} | Train: {len(X_train)} | Test: {len(X_test)}")

rf = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=10, min_samples_leaf=4,
                            max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
dt = DecisionTreeClassifier(max_depth=25, min_samples_split=10, min_samples_leaf=4,
                            class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

acc = accuracy_score(y_test, y_pred)*100
mf1 = f1_score(y_test, y_pred, average='macro')*100
print(f"  RF Accuracy: {acc:.2f}% | Macro-F1: {mf1:.2f}%")


# ================================================================
#  FIG 5.1 — DT vs RF: Per-Class Metrics Comparison
# ================================================================
print("\n  Fig 5.1: DT vs RF Per-Class Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
metric_names = ['Precision', 'Recall', 'F1-Score']
for idx, (metric_fn, mname) in enumerate(zip(
    [precision_score, recall_score, f1_score], metric_names)):
    ax = axes[idx]
    dt_vals = [metric_fn(y_test==i, dt_pred==i)*100 for i in range(3)]
    rf_vals = [metric_fn(y_test==i, y_pred==i)*100 for i in range(3)]
    x = np.arange(3)
    b1 = ax.bar(x-0.18, dt_vals, 0.34, color=PAL['dt'], edgecolor='white', lw=1.2, label='Decision Tree', zorder=3)
    b2 = ax.bar(x+0.18, rf_vals, 0.34, color=PAL['rf'], edgecolor='white', lw=1.2, label='Random Forest', zorder=3)
    for b in [b1, b2]:
        for bar in b:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+1.2, f'{h:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title(mname, fontsize=14, pad=10)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if idx == 0: ax.set_ylabel('Score (%)', fontsize=12)

axes[0].legend(fontsize=10, loc='upper right', framealpha=0.9)
fig.suptitle('Algorithm Comparison: Decision Tree vs. Random Forest (Per-Class Breakdown)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_1_dt_vs_rf.png', dpi=300, bbox_inches='tight'); plt.close()


# ================================================================
#  FIG 5.2 — Dual Confusion Matrix (Counts + Normalized)
# ================================================================
print("  Fig 5.2: Dual Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
# Raw counts
sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax1,
            xticklabels=LABELS, yticklabels=LABELS, linewidths=2, linecolor='white',
            annot_kws={'size':15,'fontweight':'bold'}, cbar_kws={'label':'Count','shrink':0.8})
ax1.set_xlabel('Predicted', fontsize=12); ax1.set_ylabel('Actual', fontsize=12)
ax1.set_title('(a) Absolute Counts', fontsize=13, fontweight='bold', pad=12)

# Normalized
sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2, vmin=0, vmax=100,
            xticklabels=LABELS, yticklabels=LABELS, linewidths=2, linecolor='white',
            annot_kws={'size':14,'fontweight':'bold'}, cbar_kws={'label':'%','shrink':0.8})
for t in ax2.texts:
    t.set_text(t.get_text() + '%')
ax2.set_xlabel('Predicted', fontsize=12); ax2.set_ylabel('Actual', fontsize=12)
ax2.set_title('(b) Row-Normalized (%)', fontsize=13, fontweight='bold', pad=12)

fig.suptitle('Confusion Matrix Analysis — Primary Model (3-Class, 62 Structural Features)',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_2_confusion_matrix.png', dpi=300, bbox_inches='tight'); plt.close()


# ================================================================
#  FIG 5.3 — Classification Report Heatmap
# ================================================================
print("  Fig 5.3: Classification Report Heatmap...")
report = classification_report(y_test, y_pred, target_names=LABELS, output_dict=True)
metrics_df = pd.DataFrame({
    'Precision': [report[l]['precision']*100 for l in LABELS],
    'Recall':    [report[l]['recall']*100 for l in LABELS],
    'F1-Score':  [report[l]['f1-score']*100 for l in LABELS],
    'Support':   [report[l]['support'] for l in LABELS]
}, index=LABELS)

fig, ax = plt.subplots(figsize=(10, 4.5))
hm_data = metrics_df[['Precision','Recall','F1-Score']]
sns.heatmap(hm_data, annot=True, fmt='.1f', cmap='RdYlGn', vmin=30, vmax=95,
            linewidths=2.5, linecolor='white', ax=ax,
            annot_kws={'size':16,'fontweight':'bold'}, cbar_kws={'label':'Score (%)'})
for t in ax.texts:
    t.set_text(t.get_text() + '%')
# Embed support counts into y-axis labels
ylabels = [f'{label}  (n={int(row["Support"]):,})' for label, row in metrics_df.iterrows()]
ax.set_yticklabels(ylabels, rotation=0, fontsize=12, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, fontweight='bold')
ax.set_title('Per-Class Performance Report — Random Forest Primary Model',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_3_classification_report.png', dpi=300, bbox_inches='tight'); plt.close()


# ================================================================
#  FIG 5.4 — Cross-Validation Stability
# ================================================================
print("  Fig 5.4: Cross-Validation Stability...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
cv_mean, cv_std = cv_scores.mean()*100, cv_scores.std()*100
print(f"    CV: {cv_mean:.2f}% +/- {cv_std:.2f}%")

fig, ax = plt.subplots(figsize=(9, 5.5))
folds = np.arange(1, 6)
bars = ax.bar(folds, cv_scores*100, width=0.55, color=[PAL['mid']]*5, edgecolor='white', lw=1.5, zorder=3)
for bar, s in zip(bars, cv_scores*100):
    ax.text(bar.get_x()+bar.get_width()/2, s+0.3, f'{s:.2f}%', ha='center', fontsize=11, fontweight='bold')

ax.axhline(cv_mean, color=PAL['accent'], ls='--', lw=2, zorder=4, label=f'Mean: {cv_mean:.2f}%')
ax.fill_between([0.4,5.6], cv_mean-cv_std, cv_mean+cv_std, alpha=0.15, color=PAL['accent'], zorder=2)
ax.set_xlabel('Fold Number', fontsize=12); ax.set_ylabel('Macro-F1 Score (%)', fontsize=12)
ax.set_xticks(folds); ax.set_xlim(0.4, 5.6)
ax.set_ylim(cv_mean-5, cv_mean+5)
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='y', alpha=0.4, zorder=0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Stats box
stats = f'Mean: {cv_mean:.2f}%\nStd: \u00b1{cv_std:.2f}%\nRange: {cv_scores.min()*100:.2f}-{cv_scores.max()*100:.2f}%'
ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=PAL['dark'], alpha=0.9))
ax.set_title('5-Fold Stratified Cross-Validation Stability', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_4_cv_stability.png', dpi=300, bbox_inches='tight'); plt.close()


# ================================================================
#  FIG 5.5 — Learning Curve
# ================================================================
print("  Fig 5.5: Learning Curve...")
rf_lc = RandomForestClassifier(n_estimators=300, max_depth=25, min_samples_split=10, min_samples_leaf=4,
                               max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
sizes, tr_sc, te_sc = learning_curve(rf_lc, X, y,
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    train_sizes=np.linspace(0.1, 1.0, 6), scoring='f1_macro', n_jobs=-1)
tr_m, tr_s = tr_sc.mean(1)*100, tr_sc.std(1)*100
te_m, te_s = te_sc.mean(1)*100, te_sc.std(1)*100

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.fill_between(sizes, tr_m-tr_s, tr_m+tr_s, alpha=0.12, color=PAL['flop'])
ax.fill_between(sizes, te_m-te_s, te_m+te_s, alpha=0.12, color=PAL['success'])
ax.plot(sizes, tr_m, 'o-', color=PAL['flop'], lw=2.5, ms=8, label='Training Score', zorder=4)
ax.plot(sizes, te_m, 's-', color=PAL['success'], lw=2.5, ms=8, label='Cross-Validation Score', zorder=4)

# Annotate final gap
gap = tr_m[-1] - te_m[-1]
ax.annotate(f'Gap: {gap:.1f}pp', xy=(sizes[-1], (tr_m[-1]+te_m[-1])/2),
            xytext=(-80, 0), textcoords='offset points', fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=PAL['dark']),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=PAL['dark']))

ax.set_xlabel('Number of Training Samples', fontsize=12)
ax.set_ylabel('Macro-F1 Score (%)', fontsize=12)
ax.set_title('Learning Curve: Performance vs. Dataset Size', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(alpha=0.4); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_5_learning_curve.png', dpi=300, bbox_inches='tight'); plt.close()


# ================================================================
#  FIG 5.6 — 4-Experiment Comparison
# ================================================================
print("  Fig 5.6: Experiment Comparison...")
exp4_acc = accuracy_score(y_test, y_pred)*100
exp4_f1 = f1_score(y_test, y_pred, average='macro')*100

names = ['EXP 1\n4-Class\n+Reviews', 'EXP 2\n4-Class\nStructural', 'EXP 3\n3-Class\n+Reviews', 'EXP 4\n3-Class\nStructural']
e_acc = [85.99, 79.73, 86.06, round(exp4_acc,2)]
e_f1 =  [75.16, 52.28, 78.04, round(exp4_f1,2)]
colors_acc = [PAL['light']]*4
colors_f1 = [PAL['dark']]*4

fig, ax = plt.subplots(figsize=(12, 6.5))
x = np.arange(4); w = 0.34
b1 = ax.bar(x-w/2, e_acc, w, color=colors_acc, edgecolor='white', lw=1.5, label='Accuracy', zorder=3)
b2 = ax.bar(x+w/2, e_f1, w, color=colors_f1, edgecolor='white', lw=1.5, label='Macro-F1', zorder=3)
for b in [b1, b2]:
    for bar in b:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9.5, fontweight='bold')

# Highlight primary model
ax.axvspan(2.6, 3.6, alpha=0.08, color=PAL['accent'], zorder=0)
ax.annotate('PRIMARY\nMODEL', xy=(3, max(e_acc[3], e_f1[3])+6), fontsize=10,
            fontweight='bold', color=PAL['accent'], ha='center')

ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel('Score (%)', fontsize=12); ax.set_ylim(0, 100)
ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.95)
ax.grid(axis='y', alpha=0.4, zorder=0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_title('Experiment Comparison: Accuracy vs. Macro-F1 Across Configurations',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_6_experiment_comparison.png', dpi=300, bbox_inches='tight'); plt.close()


# ================================================================
#  FIG 5.7 — Feature Importance Top 20 (with std from trees)
# ================================================================
print("  Fig 5.7: Feature Importance (with tree variance)...")
importances = pd.Series(rf.feature_importances_, index=X.columns)
# Get std from individual trees
tree_imps = np.array([t.feature_importances_ for t in rf.estimators_])
imp_std = pd.Series(tree_imps.std(axis=0), index=X.columns)

top20 = importances.sort_values(ascending=True).tail(20)
top20_std = imp_std[top20.index]

def cat_color(f):
    if f.startswith('tag_'): return PAL['light']
    if f.startswith('genre_'): return '#82E0AA'
    if f.startswith('lang_') and f != 'lang_count': return PAL['accent']
    if f.startswith('has_') or f.startswith('supports_'): return PAL['niche']
    return PAL['dark']

fig, ax = plt.subplots(figsize=(11, 8))
colors = [cat_color(f) for f in top20.index]
bars = ax.barh(range(len(top20)), top20.values*100,
               color=colors, edgecolor='white', height=0.72, zorder=3)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([f.replace('_',' ').title() for f in top20.index], fontsize=10)

for i, v in enumerate(top20.values*100):
    ax.text(v + 0.15, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')

legend_els = [Patch(facecolor=PAL['dark'], label='Numerical'),
              Patch(facecolor=PAL['niche'], label='Binary Flag'),
              Patch(facecolor=PAL['light'], label='Tag'),
              Patch(facecolor='#82E0AA', label='Genre'),
              Patch(facecolor=PAL['accent'], label='Language')]
ax.legend(handles=legend_els, loc='lower right', fontsize=10, framealpha=0.95)
ax.set_xlabel('Feature Importance (%)', fontsize=12)
ax.set_title('Top 20 Most Important Features',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, zorder=0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_7_feature_importance.png', dpi=300, bbox_inches='tight'); plt.close()


# ================================================================
#  FIG 5.8 — Feature Category Contribution (Donut)
# ================================================================
print("  Fig 5.8: Feature Category Contribution...")
cat_imp = {
    'Numerical': sum(importances[f] for f in ['price','achievements','lang_count','dlc_count'] if f in importances.index),
    'Binary Flags': sum(importances[f] for f in importances.index if f.startswith('has_') or f.startswith('supports_')),
    'Tags': sum(importances[f] for f in importances.index if f.startswith('tag_')),
    'Languages': sum(importances[f] for f in importances.index if f.startswith('lang_') and f != 'lang_count'),
    'Genres': sum(importances[f] for f in importances.index if f.startswith('genre_')),
}
vals = [v*100 for v in cat_imp.values()]
cols = [PAL['dark'], PAL['niche'], PAL['light'], PAL['accent'], '#82E0AA']

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(vals, labels=list(cat_imp.keys()), autopct='%1.1f%%',
    colors=cols, startangle=90, pctdistance=0.78, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2.5),
    textprops={'fontsize':12, 'fontweight':'bold'})
for t in autotexts:
    t.set_fontsize(11); t.set_fontweight('bold'); t.set_color('white')

# Center stats
ax.text(0, 0.06, f'{len(structural)}', fontsize=32, fontweight='bold', ha='center', va='center', color=PAL['dark'])
ax.text(0, -0.1, 'Features', fontsize=12, ha='center', va='center', color='#666')
ax.set_title('Feature Category Contribution to Model Decisions', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_8_category_contribution.png', dpi=300, bbox_inches='tight'); plt.close()


# ================================================================
#  FIG 5.9 — Per-Class Prediction Confidence
# ================================================================
print("  Fig 5.9: Prediction Confidence Distribution...")
y_proba = rf.predict_proba(X_test)
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for i, (ax, label, color) in enumerate(zip(axes, LABELS, CLASS_COLORS)):
    mask_correct = (y_test.values == i) & (y_pred == i)
    mask_wrong = (y_test.values == i) & (y_pred != i)
    if mask_correct.sum() > 0:
        ax.hist(y_proba[mask_correct, i], bins=25, alpha=0.7, color=color, edgecolor='white',
                label=f'Correct (n={mask_correct.sum():,})', zorder=3)
    if mask_wrong.sum() > 0:
        ax.hist(y_proba[mask_wrong, i], bins=25, alpha=0.5, color='#AAB7B8', edgecolor='white',
                label=f'Misclassified (n={mask_wrong.sum():,})', zorder=2)
    ax.set_title(label, fontsize=13, fontweight='bold', color=color)
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.legend(fontsize=9, loc='upper center')
    ax.axvline(0.5, ls=':', color='#888', lw=1)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Number of Games', fontsize=12)
fig.suptitle('Model Confidence: Predicted Probability Distribution by True Class',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_9_confidence_distribution.png', dpi=300, bbox_inches='tight'); plt.close()


# ================================================================
#  FIG 5.10 — Error Analysis: Where Does the Model Fail?
# ================================================================
print("  Fig 5.10: Error Analysis...")
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(3)
correct = np.diag(cm_pct)
misclass = 100 - correct

b1 = ax.bar(x, correct, 0.5, color=CLASS_COLORS, edgecolor='white', lw=1.5, label='Correct', zorder=3)
b2 = ax.bar(x, misclass, 0.5, bottom=correct, color=['#FADBD8','#D6EAF8','#D5F5E3'],
            edgecolor='white', lw=1.5, label='Misclassified', zorder=3)

for i in range(3):
    ax.text(i, correct[i]/2, f'{correct[i]:.1f}%', ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    if misclass[i] > 5:
        # Show error breakdown with vertical spacing
        err_lines = []
        for j in range(3):
            if i != j and cm_pct[i][j] > 3:
                err_lines.append(f'{cm_pct[i][j]:.0f}% \u2192 {LABELS[j]}')
        err_text = '\n'.join(err_lines)
        ax.text(i, correct[i]+misclass[i]/2, f'{misclass[i]:.1f}%\n{err_text}',
                ha='center', va='center', fontsize=9, fontweight='bold', color='#555')

ax.set_xticks(x); ax.set_xticklabels([f'{l}\n(n={cm.sum(axis=1)[i]:,})' for i, l in enumerate(LABELS)], fontsize=11)
ax.set_ylabel('Percentage (%)', fontsize=12); ax.set_ylim(0, 108)
ax.set_title('Error Analysis: Classification Accuracy by Class', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_10_error_analysis.png', dpi=300, bbox_inches='tight'); plt.close()


# ================================================================
print(f"\n{'='*60}")
print(f"  DONE — 10 thesis-grade figures saved to {OUT}/")
print(f"{'='*60}")
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(f'{OUT}/{f}')/1024
    print(f"    {f} ({sz:.0f} KB)")
