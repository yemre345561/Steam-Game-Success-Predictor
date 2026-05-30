"""
=============================================================================
  RELEASE_YEAR REMOVAL - UPDATED FIGURES ONLY
=============================================================================
  Bu script sadece release_year'in cikarilmasindan etkilenen 3 grafigi
  yeniden uretir. Orijinal dosyalara DOKUNMAZ.

  Uretilen grafikler:
    degisen_grafikler/
      fig3_1_feature_anatomy.png       -> 59 -> 58, Numerical 5 -> 4
      fig3_2_dimensionality_expansion.png -> Other Metadata 16 -> 15
      fe_02_target_correlation_rank.png   -> Release Year bari cikarildi
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
#  OUTPUT: Ayrı klasör - eski dosyalara dokunmuyoruz
# ============================================================================
OUT_DIR = 'degisen_grafikler'
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("  RELEASE_YEAR REMOVAL - UPDATED FIGURES GENERATOR")
print("=" * 70)

# ============================================================================
#  COMMON STYLING
# ============================================================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

C3 = {
    'primary': '#1B2A4A', 'secondary': '#2E5090', 'accent': '#4A90D9',
    'light': '#7AB8F5', 'highlight': '#E8A838',
    'flop': '#C0392B', 'niche': '#2E86C1', 'success': '#27AE60'
}

COLORS = {
    'primary': '#2C3E50', 'secondary': '#34495E', 'accent': '#E67E22',
    'light': '#BDC3C7', 'highlight': '#F1C40F',
    'flop': '#E74C3C', 'niche': '#3498DB', 'success': '#2ECC71'
}


# ////////////////////////////////////////////////////////////////////////////
#  FIGURE 1: Feature Anatomy (59 -> 58, Numerical 5 -> 4)
# ////////////////////////////////////////////////////////////////////////////
print("\n>>> Fig 3.1: Feature Space Anatomy (UPDATED: 62 features)...")
fig, ax = plt.subplots(figsize=(12, 4))

# Updated: 7 Numerical, 12 Binary, 8 Lang, 10 Genre, 25 Tags = 62
cat_names = ['Numerical\n(7 Feat.)', 'Binary Flags\n(12 Feat.)', 'Languages\n(8 Feat.)', 'Genres\n(10 Feat.)', 'Tags\n(25 Feat.)']
cat_counts = [7, 12, 8, 10, 25]  # Total: 62
cat_colors = [C3['primary'], C3['secondary'], C3['highlight'], C3['light'], C3['accent']]

left = 0
for name, count, color in zip(cat_names, cat_counts, cat_colors):
    ax.barh('Final Input Vector', count, left=left, color=color, edgecolor='white', height=0.6, label=name)
    textColor = 'white' if color in [C3['primary'], C3['secondary'], C3['accent']] else 'black'
    ax.text(left + count/2, 0, f'{name}', ha='center', va='center', color=textColor, fontweight='bold', fontsize=11)
    left += count

ax.set_title(f'Final Feature Engineering Result: Anatomy of {sum(cat_counts)} Predictive Variables', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Total Number of Features in the Matrix', fontsize=13)
ax.set_xlim(0, sum(cat_counts))
ax.set_yticks([])
sns.despine(left=True, top=True, right=True)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_1_feature_anatomy.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> SAVED: fig3_1_feature_anatomy.png")


# ////////////////////////////////////////////////////////////////////////////
#  FIGURE 2: Dimensionality Expansion (Other Metadata 16 -> 15)
# ////////////////////////////////////////////////////////////////////////////
print("\n>>> Fig 3.2: Dimensionality Expansion (UPDATED: Other Metadata 15)...")
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Languages', 'Genres', 'Tags', 'Other Metadata']
raw_counts = [1, 1, 1, 44]  # Raw input columns (unchanged)
# Updated: Other metadata = 7 Num + 12 Binary = 19
# Total = 8 + 10 + 25 + 19 = 62
eng_counts = [8, 10, 25, 19]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, raw_counts, width, label='Raw Input Columns', color=C3['flop'], edgecolor='white')
bars2 = ax.bar(x + width/2, eng_counts, width, label='Engineered AI Matrix', color=C3['success'], edgecolor='white')

ax.bar_label(bars1, padding=3, fontsize=12, fontweight='bold')
ax.bar_label(bars2, padding=3, fontsize=12, fontweight='bold')

ax.set_title('Feature Engineering Impact: Dimensionality Expansion via Encodings', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Number of Columns in Dataset', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=13)
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.set_ylim(0, 55)

sns.despine(top=True, right=True)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_2_dimensionality_expansion.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> SAVED: fig3_2_dimensionality_expansion.png")


# ////////////////////////////////////////////////////////////////////////////
#  FIGURE 3: Pearson Correlation Rank (release_year bari cikarildi)
# ////////////////////////////////////////////////////////////////////////////
print("\n>>> Fe_02: Pearson Correlation Rank (UPDATED: Release Year removed)...")
print("  Loading model_ready_dataset.csv...")
df = pd.read_csv('model_ready_dataset.csv')
df['target_3class'] = df['target_tier'].replace({3: 2})

# Feature subsets - release_year HARIC
features_to_exclude = ['target_tier', 'target_3class', 'peak_ccu', 'pct_pos_total', 'num_reviews_total', 'release_year']
structural_features = [c for c in df.columns if c not in features_to_exclude]

# Calculate correlations (release_year artik structural_features'ta yok)
correlations = {}
for feat in structural_features:
    corr_val = df[feat].corr(df['target_3class'])
    if not pd.isna(corr_val):
        correlations[feat] = corr_val

corr_series = pd.Series(correlations).sort_values(ascending=True)

top_pos = corr_series.tail(15)
top_neg = corr_series.head(10)

plot_data = pd.concat([top_neg, top_pos])

# Reset style for this chart
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12, 'figure.dpi': 300,
    'axes.titleweight': 'bold', 'axes.labelweight': 'bold',
    'figure.facecolor': '#FFFFFF', 'axes.facecolor': '#F8F9FA'
})

fig, ax = plt.subplots(figsize=(10, 8))

colors = [COLORS['flop'] if v < 0 else COLORS['success'] for v in plot_data.values]
bars = ax.barh(plot_data.index.str.replace('_', ' ').str.title(), plot_data.values, color=colors)

ax.set_title('Strongest Structural Predictors of Game Success\n(Pearson Correlation with 3-Class Target Tier)', pad=20)
ax.set_xlabel('Correlation Coefficient (r)')

for i, (bar, val) in enumerate(zip(bars, plot_data.values)):
    align = 'right' if val < 0 else 'left'
    offset = -0.004 if val < 0 else 0.004
    ax.text(val + offset, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', ha=align, fontweight='bold', fontsize=9)

ax.axvline(0, color='black', linewidth=1)
sns.despine(left=True, bottom=False)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fe_02_target_correlation_rank.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> SAVED: fe_02_target_correlation_rank.png")


# ============================================================================
#  SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("  TAMAMLANDI - 3 GUNCELLENMIS GRAFIK URETILDI")
print("=" * 70)
print(f"\n  Cikti klasoru: ./{OUT_DIR}/")

files = sorted(os.listdir(OUT_DIR))
for f in files:
    size = os.path.getsize(f"{OUT_DIR}/{f}") / 1024
    print(f"    - {f} ({size:.0f} KB)")

print(f"\n  ORIJINAL dosyalara DOKUNULMADI.")
print(f"  Begenirsen 'tez bolum 3 resimler' ve 'tez bolum 4 resimler' klasorlerine")
print(f"  elle kopyalayabilirsin.")
