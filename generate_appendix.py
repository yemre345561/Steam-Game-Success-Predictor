"""
=============================================================================
  APPENDIX TABLE GENERATOR
=============================================================================
  Generates 3 appendix tables as high-quality PNG images.
  
  Output:
    appendix/
      appendix_a_feature_dictionary.png
      appendix_b_hyperparameter_grid.png
      appendix_c_tag_divergence.png
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import ast, os, warnings
warnings.filterwarnings('ignore')

OUT_DIR = 'appendix'
os.makedirs(OUT_DIR, exist_ok=True)

# Color scheme matching thesis
HEADER_COLOR = '#1B2A4A'
ROW_EVEN = '#F8F9FA'
ROW_ODD = '#FFFFFF'
ACCENT = '#2E5090'

print("=" * 70)
print("  APPENDIX TABLE GENERATOR")
print("=" * 70)

# ============================================================================
#  APPENDIX A: COMPLETE FEATURE DICTIONARY
# ============================================================================
print("\n>>> Generating Appendix A: Complete Feature Dictionary...")

df = pd.read_csv('model_ready_dataset.csv')
review_metrics = ['pct_pos_total', 'num_reviews_total', 'peak_ccu']
all_features = [c for c in df.columns if c not in ['target_tier', 'target_3class']]
structural_features = [c for c in all_features if c not in review_metrics + ['release_year']]

# Categorize and describe each feature
feature_info = []
for f in structural_features:
    # Determine category
    # Special cases: lang_count and tag_count are numerical, not binary
    if f == 'lang_count' or f == 'tag_count':
        cat = 'Numerical'
        desc = {'lang_count': 'Total number of supported languages',
                'tag_count': 'Total number of user-applied Steam tags'}[f]
    elif f.startswith('lang_'):
        cat = 'Language'
        lang_name = f.replace('lang_', '').replace('_', ' ').title()
        desc = f'Binary flag: 1 if game supports {lang_name} localization'
    elif f.startswith('genre_'):
        cat = 'Genre'
        genre_name = f.replace('genre_', '').replace('_', ' ').title()
        desc = f'Binary flag: 1 if game is categorized as {genre_name}'
    elif f.startswith('tag_'):
        cat = 'Tag'
        tag_name = f.replace('tag_', '').replace('_', ' ').title()
        desc = f'Binary flag: 1 if users tagged this game as {tag_name}'
    elif f.startswith('is_') or f.startswith('has_') or f.startswith('supports_'):
        cat = 'Binary Flag'
        descriptions = {
            'is_free': 'Whether the game uses a free-to-play monetization model',
            'is_early_access': 'Whether the game is released under Early Access',
            'has_achievements': 'Whether the game offers Steam Achievements',
            'has_dlc': 'Whether the game has downloadable content available',
            'has_multiplayer': 'Whether the game supports multiplayer gameplay',
            'has_coop': 'Whether the game supports cooperative gameplay',
            'has_controller': 'Whether the game supports controller input',
            'has_trading_cards': 'Whether the game offers Steam Trading Cards',
            'has_workshop': 'Whether the game supports Steam Workshop modding',
            'has_website': 'Whether the developer maintains an official website',
            'supports_mac': 'Whether the game is compatible with macOS',
            'supports_linux': 'Whether the game is compatible with Linux',
        }
        desc = descriptions.get(f, f'Binary indicator for {f}')
    else:
        cat = 'Numerical'
        descriptions = {
            'price': 'Base retail price of the game in USD',
            'achievements': 'Total number of Steam Achievements offered',
            'screenshot_count': 'Number of screenshots on the store page',
            'required_age': 'Minimum age rating required to access the game',
            'lang_count': 'Total number of supported languages',
            'tag_count': 'Total number of user-applied Steam tags',
            'dlc_count': 'Total number of downloadable content packages',
        }
        desc = descriptions.get(f, f'Numerical value for {f}')
    
    feature_info.append([f, cat, desc])

# Sort by category order
cat_order = {'Numerical': 0, 'Binary Flag': 1, 'Language': 2, 'Genre': 3, 'Tag': 4}
feature_info.sort(key=lambda x: (cat_order.get(x[1], 5), x[0]))

# Create table image - split into pages if needed
n_features = len(feature_info)
rows_per_page = 35
n_pages = (n_features + rows_per_page - 1) // rows_per_page

for page in range(n_pages):
    start = page * rows_per_page
    end = min(start + rows_per_page, n_features)
    page_data = feature_info[start:end]
    
    fig, ax = plt.subplots(figsize=(16, len(page_data) * 0.45 + 1.0))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    if page == 0:
        title = f'Complete Feature Dictionary ({n_features} Structural Features)'
    else:
        title = 'Complete Feature Dictionary (Continued)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10, color=HEADER_COLOR)
    
    col_labels = ['#', 'Feature Name', 'Category', 'Description']
    table_data = [[str(start + i + 1), row[0], row[1], row[2]] for i, row in enumerate(page_data)]
    
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='left',
        loc='upper center',
        colWidths=[0.04, 0.22, 0.12, 0.62]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    
    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_edgecolor('white')
    
    # Style rows with alternating colors
    cat_colors = {
        'Numerical': '#E8F4FD',
        'Binary Flag': '#FFF3E0',
        'Language': '#E8F5E9',
        'Genre': '#F3E5F5',
        'Tag': '#FFF8E1'
    }
    
    for i, row in enumerate(page_data):
        cat = row[1]
        bg_color = cat_colors.get(cat, ROW_EVEN) if i % 2 == 0 else ROW_ODD
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor(bg_color)
            cell.set_edgecolor('#E0E0E0')
    
    plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.01)
    
    # Render and resize to crop whitespace below table
    fig.canvas.draw()
    table_bbox = table.get_window_extent(fig.canvas.get_renderer())
    title_bbox = ax.title.get_window_extent(fig.canvas.get_renderer())
    # Combine table + title bounding boxes
    from matplotlib.transforms import Bbox
    combined = Bbox.union([table_bbox, title_bbox])
    # Convert to figure fraction and add small padding
    fig_bbox = combined.transformed(fig.transFigure.inverted())
    new_height = fig.get_figheight() * (fig_bbox.y1 - fig_bbox.y0 + 0.04)
    fig.set_figheight(max(new_height, 2.0))
    
    suffix = f'_page{page+1}' if n_pages > 1 else ''
    plt.savefig(f'{OUT_DIR}/appendix_a_feature_dictionary{suffix}.png', dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close()
    print(f"  Saved Appendix A page {page+1}/{n_pages}")

# ============================================================================
#  APPENDIX B: HYPERPARAMETER OPTIMIZATION GRID
# ============================================================================
print("\n>>> Generating Appendix B: Hyperparameter Optimization Grid...")

fig, ax = plt.subplots(figsize=(14, 3.8))
ax.axis('off')
ax.set_title('Hyperparameter Optimization Grid (GridSearchCV)',
             fontsize=14, fontweight='bold', pad=6, color=HEADER_COLOR)

col_labels = ['Hyperparameter', 'Tested Values', 'Selected Optimal', 'Rationale']
table_data = [
    ['n_estimators', '200, 400', '300*', 'Ensemble breadth for prediction stability'],
    ['max_depth', 'None, 30, 50', '25*', 'Limits tree complexity to prevent memorization'],
    ['min_samples_split', '2, 5, 10', '10', 'Ensures splits only at statistically meaningful clusters'],
    ['min_samples_leaf', '1, 2, 4', '4', 'Prevents leaf nodes from isolating outlier games'],
    ['max_features', 'sqrt (fixed)', 'sqrt', 'Decorrelates individual trees in the ensemble'],
    ['class_weight', 'balanced', 'balanced', 'Penalizes misclassification of minority success tiers'],
    ['random_state', '42', '42', 'Ensures full reproducibility of results'],
    ['cv (folds)', '3', '3', 'Cross-validation folds during grid search'],
    ['scoring', 'f1_macro', 'f1_macro', 'Optimizes for balanced multi-class performance'],
]

# Footnote removed - user will add it as text in the Word document

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='left',
    loc='upper center',
    colWidths=[0.18, 0.18, 0.16, 0.48]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor(HEADER_COLOR)
    cell.set_text_props(color='white', fontweight='bold', fontsize=11)
    cell.set_edgecolor('white')

# Style rows
for i in range(len(table_data)):
    bg = ROW_EVEN if i % 2 == 0 else ROW_ODD
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor('#E0E0E0')
        if j == 2:  # Highlight selected column
            cell.set_text_props(fontweight='bold', color=ACCENT)

# Footnote text removed - will be added in Word document

plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
plt.savefig(f'{OUT_DIR}/appendix_b_hyperparameter_grid.png', dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
plt.close()
print("  Saved Appendix B")

# ============================================================================
#  APPENDIX C: TAG SELECTION DIVERGENCE ANALYSIS
# ============================================================================
print("\n>>> Generating Appendix C: Tag Selection Divergence Analysis...")

# Tag divergence is calculated from the processed dataset (already loaded as df)
# Exclude tag_count as it is a numerical feature (total tag count), not a binary tag indicator
tag_features = [f for f in structural_features if f.startswith('tag_') and f != 'tag_count']

# Calculate divergence for each tag
df['target_3class'] = df['target_tier'].replace({3: 2})
flop_mask = df['target_3class'] == 0
success_mask = df['target_3class'] == 2

tag_divergence = []
for tag in tag_features:
    flop_pct = df.loc[flop_mask, tag].mean() * 100
    success_pct = df.loc[success_mask, tag].mean() * 100
    divergence = abs(success_pct - flop_pct)
    direction = '↑ Success' if success_pct > flop_pct else '↓ Flop'
    tag_divergence.append([
        tag.replace('tag_', '').replace('_', ' ').title(),
        f'{flop_pct:.1f}%',
        f'{success_pct:.1f}%',
        f'{divergence:.1f}pp',
        direction
    ])

# Sort by divergence (descending)
tag_divergence.sort(key=lambda x: float(x[3].replace('pp', '')), reverse=True)

# Add rank
for i, row in enumerate(tag_divergence):
    row.insert(0, str(i + 1))

fig, ax = plt.subplots(figsize=(15, len(tag_divergence) * 0.45 + 1.0))
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Tag Selection Divergence Analysis (Top 25 Gameplay Tags)',
             fontsize=14, fontweight='bold', pad=10, color=HEADER_COLOR)

col_labels = ['Rank', 'Tag Name', 'Flop Prevalence', 'Successful Prevalence', 'Absolute Divergence', 'Direction']
table = ax.table(
    cellText=tag_divergence,
    colLabels=col_labels,
    cellLoc='center',
    loc='upper center',
    colWidths=[0.06, 0.22, 0.16, 0.18, 0.18, 0.14]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.4)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor(HEADER_COLOR)
    cell.set_text_props(color='white', fontweight='bold', fontsize=10)
    cell.set_edgecolor('white')

# Style rows with color coding
for i, row in enumerate(tag_divergence):
    divergence_val = float(row[4].replace('pp', ''))
    direction = row[5]
    
    if divergence_val >= 20:
        bg = '#E8F5E9' if direction == '↑ Success' else '#FFEBEE'
    elif divergence_val >= 10:
        bg = '#F1F8E9' if direction == '↑ Success' else '#FFF3E0'
    else:
        bg = ROW_EVEN if i % 2 == 0 else ROW_ODD
    
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor('#E0E0E0')
        if j == 4:  # Highlight divergence
            cell.set_text_props(fontweight='bold')

# Footnote removed - user will add it as text in the Word document

plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.01)

# Render and resize to crop whitespace below table
fig.canvas.draw()
table_bbox = table.get_window_extent(fig.canvas.get_renderer())
title_bbox = ax.title.get_window_extent(fig.canvas.get_renderer())
from matplotlib.transforms import Bbox
combined = Bbox.union([table_bbox, title_bbox])
fig_bbox = combined.transformed(fig.transFigure.inverted())
new_height = fig.get_figheight() * (fig_bbox.y1 - fig_bbox.y0 + 0.04)
fig.set_figheight(max(new_height, 2.0))

plt.savefig(f'{OUT_DIR}/appendix_c_tag_divergence.png', dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
plt.close()
print("  Saved Appendix C")

print("\n" + "=" * 70)
print("  ALL APPENDIX TABLES GENERATED SUCCESSFULLY")
print(f"  Output directory: {OUT_DIR}/")
print("=" * 70)
