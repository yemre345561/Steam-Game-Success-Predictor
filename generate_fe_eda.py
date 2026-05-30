"""
=============================================================================
  STEAM SUCCESS PREDICTION - FEATURE ENGINEERING EDA SCRIPT
=============================================================================
  Generates deeper analytical insights on the extracted/engineered features
  (One-hot tags, dummy genres, binary language flags, normalized numericals).
  This acts as the bridge between raw data EDA (Chapter 4) and ML Results (Chapter 5).
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.stats import pointbiserialr
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

# Styling Setup
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12, 
    'figure.dpi': 300,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#F8F9FA'
})

COLORS = {
    'primary': '#2C3E50',
    'secondary': '#34495E', 
    'accent': '#E67E22',
    'light': '#BDC3C7',
    'highlight': '#F1C40F',
    'flop': '#E74C3C',
    'niche': '#3498DB',
    'success': '#2ECC71'
}

OUT_DIR = "chapter4_feature_engineering_eda"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load Data
print("Loading model_ready_dataset.csv...")
df = pd.read_csv('model_ready_dataset.csv')

# Convert 4-tier to our 3-class target variable for structural alignment with ML model
df['target_3class'] = df['target_tier'].replace({3: 2})
# 0 = Flop (<20K), 1 = Niche (20K-200K), 2 = Successful (>200K)

# Define feature subsets (Strictly Structural, NO Review metrics to avoid leakage)
features_to_exclude = ['target_tier', 'target_3class', 'peak_ccu', 'pct_pos_total', 'num_reviews_total', 'release_year']
structural_features = [c for c in df.columns if c not in features_to_exclude]

tags = [f for f in structural_features if f.startswith('tag_') and f != 'tag_count']
genres = [f for f in structural_features if f.startswith('genre_')]
langs = [f for f in structural_features if f.startswith('lang_') and f != 'lang_count']
binaries = [f for f in structural_features if (f.startswith('has_') or f.startswith('supports_') or f == 'is_free')]
numericals = ['price', 'dlc_count', 'achievements', 'lang_count', 'tag_count', 'screenshot_count', 'required_age']

# Open Report File
report_path = "feature_engineering_eda_report.txt"
with open(report_path, "w", encoding="utf-8") as rf:
    def write_log(text):
        print(text)
        rf.write(text + "\n")

    write_log("================================================================================")
    write_log("  FEATURE ENGINEERING EDA REPORT ")
    write_log("  Dataset: Structural Pre-Launch Variables Only (No Data Leakage)")
    write_log("================================================================================\n")
    
    write_log(f"Total Games Analyzed: {len(df):,}")
    write_log(f"Structural Features Used: {len(structural_features)}")
    
    # --- CHART 1: Feature Density (Sparsity Analysis) ---
    write_log("\n>>> GENERATING CHART 1: Feature Density (Sparsity of Encodings)...")
    
    df['active_tags'] = df[tags].sum(axis=1)
    df['active_genres'] = df[genres].sum(axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(df['lang_count'], bins=20, ax=axes[0], color=COLORS['primary'], kde=True)
    axes[0].set_title('Supported Languages per Game', pad=15)
    axes[0].set_xlabel('Number of Languages')
    axes[0].set_ylabel('Frequency')
    
    sns.histplot(df['active_genres'], bins=10, ax=axes[1], color=COLORS['niche'], discrete=True)
    axes[1].set_title('Active Genres per Game (from OHE)', pad=15)
    axes[1].set_xlabel('Number of Genres')
    axes[1].set_ylabel('Frequency')
    
    sns.histplot(df['active_tags'], bins=25, ax=axes[2], color=COLORS['accent'], discrete=True)
    axes[2].set_title('Active Tags per Game (from OHE)', pad=15)
    axes[2].set_xlabel('Number of Top-25 Tags')
    axes[2].set_ylabel('Frequency')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fe_01_feature_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    write_log(f"  - Average Languages per game: {df['lang_count'].mean():.2f}")
    write_log(f"  - Average Genres per game:    {df['active_genres'].mean():.2f}")
    write_log(f"  - Average Top Tags per game:  {df['active_tags'].mean():.2f}")


    # --- CHART 2: Target Correlation Rank ---
    write_log("\n>>> GENERATING CHART 2: Target Correlation Ranking...")
    
    # Calculate point biserial correlation for features vs target (ordinal: 0,1,2)
    # Using simple Pearson correlation which is equivalent here for continuous vs ordinal
    correlations = {}
    for feat in structural_features:
        corr_val = df[feat].corr(df['target_3class'])
        if not pd.isna(corr_val):
            correlations[feat] = corr_val
            
    corr_series = pd.Series(correlations).sort_values(ascending=True)
    
    top_pos = corr_series.tail(15)
    top_neg = corr_series.head(10)
    
    # Plot top positive and negative predictors
    plot_data = pd.concat([top_neg, top_pos])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [COLORS['flop'] if v < 0 else COLORS['success'] for v in plot_data.values]
    bars = ax.barh(plot_data.index.str.replace('_', ' ').str.title(), plot_data.values, color=colors)
    
    ax.set_title('Strongest Structural Predictors of Game Success\n(Pearson Correlation with 3-Class Target Tier)', pad=20)
    ax.set_xlabel('Correlation Coefficient (r)')
    
    # Add values at the end of bars
    for i, (bar, val) in enumerate(zip(bars, plot_data.values)):
        align = 'right' if val < 0 else 'left'
        offset = -0.01 if val < 0 else 0.01
        ax.text(val + offset, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha=align, fontweight='bold', fontsize=9)
    
    ax.axvline(0, color='black', linewidth=1)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fe_02_target_correlation_rank.png', dpi=300, bbox_inches='tight')
    plt.close()

    write_log(f"\n  Top 5 Positive Predictors:")
    for k, v in top_pos.tail(5)[::-1].items(): write_log(f"    + {k}: {v:.3f}")
    write_log(f"  Top 5 Negative Predictors:")
    for k, v in top_neg.head(5).items(): write_log(f"    - {k}: {v:.3f}")


    # --- CHART 3: Feature Architecture Overlap (Heatmap) ---
    write_log("\n>>> GENERATING CHART 3: Feature Clusters (Heatmap)...")
    
    # Select a mix of highly correlated binary features to show clustering/overlap
    selected_features = list(top_pos.tail(10).index) + list(top_neg.head(5).index)
    selected_features = [f for f in selected_features if f not in numericals] # Keep only binaries
    
    corr_matrix = df[selected_features].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.5, vmin=-0.5, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=[c.replace('_', ' ').title() for c in corr_matrix.columns],
                yticklabels=[c.replace('_', ' ').title() for c in corr_matrix.index])
    
    ax.set_title('Collinearity & Feature Clusters Among Top Predictors', pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fe_03_genre_tags_overlap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # --- CHART 4: Class Profiles (Numerical Differentiation) ---
    write_log("\n>>> GENERATING CHART 4: Class Profiles...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    tier_labels = {0: 'Flop', 1: 'Niche', 2: 'Success'}
    df['tier_name'] = df['target_3class'].map(tier_labels)
    order = ['Flop', 'Niche', 'Success']
    pal = [COLORS['flop'], COLORS['niche'], COLORS['success']]
    
    # Price
    sns.boxplot(data=df[df['price'] <= 40], x='tier_name', y='price', order=order, ax=axes[0], palette=pal, showfliers=False)
    axes[0].set_title('Price Distribution by Tier\n(Outliers Removed)')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Price (USD)')
    
    # Lang Count
    sns.violinplot(data=df, x='tier_name', y='lang_count', order=order, ax=axes[1], palette=pal, inner='quartile')
    axes[1].set_title('Language Coverage by Tier')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Total Supported Languages')
    
    # Achievements (log scale)
    sns.boxplot(data=df[df['achievements'] > 0], x='tier_name', y='achievements', order=order, ax=axes[2], palette=pal, showfliers=False)
    axes[2].set_yscale('log')
    axes[2].set_title('Achievements Density by Tier\n(Log Scale, >0 only)')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('Achievements Count')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fe_04_class_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    for t in [0, 1, 2]:
        sub = df[df['target_3class'] == t]
        write_log(f"  Profile [{tier_labels[t]}]: Median Price=${sub['price'].median():.2f}, Median Langs={sub['lang_count'].median():.1f}")


    # --- CHART 5: Localization Dominance ---
    write_log("\n>>> GENERATING CHART 5: Localization Dominance...")
    
    lang_stats = []
    for l in langs:
        clean_name = l.replace('lang_', '').replace('_', ' ').title()
        for t in [0, 1, 2]:
            sub = df[df['target_3class'] == t]
            support_rate = sub[l].mean() * 100
            lang_stats.append({'Language': clean_name, 'Tier': tier_labels[t], 'Support Rate (%)': support_rate})
            
    lang_df = pd.DataFrame(lang_stats)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=lang_df, x='Language', y='Support Rate (%)', hue='Tier', 
                palette=pal, edgecolor='white', linewidth=1)
                
    ax.set_title('Strategic Market Localization: Support Rates Across Success Tiers', pad=20)
    ax.set_xlabel('Engineered Language Flag')
    ax.legend(title='Game Fate')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fe_05_market_dominance.png', dpi=300, bbox_inches='tight')
    plt.close()

    write_log("\n================================================================================")
    write_log("  EDA GENERATION COMPLETE")
    write_log(f"  All charts saved to: ./{OUT_DIR}/")
    write_log("================================================================================")

print("Script generation finish log written successfully.")
