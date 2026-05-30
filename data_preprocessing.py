"""
Steam Success Prediction - Data Preprocessing Pipeline
=======================================================
Transforms raw games_march2025_cleaned.csv into model_ready_dataset.csv
following the 9-step methodology defined in the implementation plan.

Academic References:
- De Luisa et al. (2021): High-cardinality feature pruning
- Wang et al. (2020) & Lu (2025): Proxy thresholding for target variable
- Golding et al. (2022): Commercial Hit threshold (~255K)
- Semenchenko (2025): AAA/Mega Hit economics
- Ma (2025) & Shovo et al. (2023): One-Hot Encoding for tree-based models
"""

import pandas as pd
import numpy as np
import ast
import json
import re
from datetime import datetime

INPUT_FILE = 'games_march2025_cleaned.csv'
OUTPUT_FILE = 'model_ready_dataset.csv'
REPORT_FILE = 'preprocessing_report.txt'

report_lines = []
def log(msg):
    print(msg)
    report_lines.append(msg)

# ═══════════════════════════════════════════════════════════════
#  LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════
log("=" * 70)
log("  STEAM DATA PREPROCESSING PIPELINE")
log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)

df = pd.read_csv(INPUT_FILE)
log(f"\nLoaded {INPUT_FILE}: {len(df)} rows x {len(df.columns)} columns")

# Save name/appid mapping for future reference
id_map = df[['appid', 'name']].copy()

# ═══════════════════════════════════════════════════════════════
#  STEP 1: DROP IRRELEVANT COLUMNS
# ═══════════════════════════════════════════════════════════════
log("\n--- STEP 1: Dropping Irrelevant Columns ---")

cols_to_drop = [
    # Near-empty columns
    'score_rank',           # 99.96% missing
    'metacritic_score',     # 96% zero
    'metacritic_url',       # 96% missing
    'pct_pos_recent',       # 92.5% sentinel
    'num_reviews_recent',   # 92.5% sentinel
    'reviews',              # 88% missing
    'notes',                # 81% missing
    # URL / media columns (processed into features below)
    'support_url', 'support_email',
    'header_image',
    # Raw text columns (would need NLP, out of scope)
    'detailed_description', 'about_the_game', 'short_description',
    # High-cardinality identifiers (De Luisa, 2021)
    'developers', 'publishers',
    # Row identifiers (saved separately)
    'appid', 'name',
    # Other non-predictive columns
    'packages',
    'user_score',           # redundant with pct_pos_total
    'recommendations',      # redundant with num_reviews_total
    'discount',             # snapshot-dependent, not a game feature
    # Playtime columns (post-launch, can leak target)
    'average_playtime_forever', 'average_playtime_2weeks',
    'median_playtime_forever', 'median_playtime_2weeks',
]

existing_drops = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_drops)
log(f"  Dropped {len(existing_drops)} columns: {existing_drops}")
log(f"  Remaining columns: {len(df.columns)}")

# ═══════════════════════════════════════════════════════════════
#  STEP 2: FIX SENTINEL VALUES (-1)
# ═══════════════════════════════════════════════════════════════
log("\n--- STEP 2: Fixing Sentinel Values ---")

sentinel_count_pct = (df['pct_pos_total'] == -1).sum()
sentinel_count_rev = (df['num_reviews_total'] == -1).sum()
log(f"  pct_pos_total sentinel (-1) count: {sentinel_count_pct}")
log(f"  num_reviews_total sentinel (-1) count: {sentinel_count_rev}")

# NOTE: has_reviews (num_reviews_total != -1) was REMOVED in the final model
# because it is derived from post-launch review data, creating indirect leakage.
# See model audit documentation for details.

# Fix sentinels
df.loc[df['pct_pos_total'] == -1, 'pct_pos_total'] = 0
df.loc[df['num_reviews_total'] == -1, 'num_reviews_total'] = 0

log(f"  Replaced all -1 sentinel values with 0")

# ═══════════════════════════════════════════════════════════════
#  STEP 3: CREATE TARGET VARIABLE (4-TIER SUCCESS MODEL)
# ═══════════════════════════════════════════════════════════════
log("\n--- STEP 3: Creating Target Variable (4 Tiers) ---")

def parse_owners(owner_str):
    """Extract lower bound from 'X - Y' estimated_owners string."""
    if pd.isna(owner_str):
        return 0
    try:
        lower = str(owner_str).split('-')[0].strip().replace(',', '')
        return int(lower)
    except:
        return 0

df['owners_lower'] = df['estimated_owners'].apply(parse_owners)

def assign_tier(owners):
    if owners < 20_000:
        return 0  # Flop
    elif owners < 200_000:
        return 1  # Niche Hit
    elif owners < 5_000_000:
        return 2  # Commercial Hit
    else:
        return 3  # Mega Hit

df['target_tier'] = df['owners_lower'].apply(assign_tier)

tier_counts = df['target_tier'].value_counts().sort_index()
tier_labels = {0: 'Flop (<20K)', 1: 'Niche Hit (20K-200K)', 2: 'Commercial Hit (200K-5M)', 3: 'Mega Hit (>=5M)'}
log("  Tier Distribution:")
for tier_id, count in tier_counts.items():
    pct = count / len(df) * 100
    log(f"    Tier {tier_id} ({tier_labels[tier_id]}): {count:,} ({pct:.2f}%)")

# ═══════════════════════════════════════════════════════════════
#  STEP 4: NUMERICAL FEATURES
# ═══════════════════════════════════════════════════════════════
log("\n--- STEP 4: Extracting Numerical Features ---")

# Extract release_year from release_date
def extract_year(date_str):
    try:
        for fmt in ['%b %d, %Y', '%B %d, %Y', '%d %b, %Y', '%Y-%m-%d']:
            try:
                return pd.to_datetime(date_str, format=fmt).year
            except:
                continue
        # Fallback: find 4-digit year
        match = re.search(r'(19|20)\d{2}', str(date_str))
        return int(match.group()) if match else np.nan
    except:
        return np.nan

df['release_year'] = df['release_date'].apply(extract_year)
df = df.drop(columns=['release_date'])

# Count supported languages
def count_languages(lang_str):
    if pd.isna(lang_str) or lang_str == '[]':
        return 0
    try:
        langs = ast.literal_eval(lang_str)
        return len(langs) if isinstance(langs, list) else 1
    except:
        # Count comma-separated entries in raw string
        return len(str(lang_str).split(','))

df['lang_count'] = df['supported_languages'].apply(count_languages)

numerical_features = ['price', 'achievements', 'dlc_count', 'release_year', 'peak_ccu', 'lang_count', 'required_age']
log(f"  Numerical features retained: {numerical_features}")
for feat in numerical_features:
    if feat in df.columns:
        log(f"    {feat}: min={df[feat].min()}, median={df[feat].median()}, max={df[feat].max()}, nulls={df[feat].isna().sum()}")

# ═══════════════════════════════════════════════════════════════
#  STEP 5: BINARY FEATURES (12 FLAGS)
# ═══════════════════════════════════════════════════════════════
log("\n--- STEP 5: Creating 12 Binary Features ---")

df['is_free'] = (df['price'] == 0).astype(int)
df['has_achievements'] = (df['achievements'] > 0).astype(int)
df['has_dlc'] = (df['dlc_count'] > 0).astype(int)

# Early Access from genres
df['is_early_access'] = df['genres'].astype(str).str.contains('Early Access', case=False, na=False).astype(int)

# Multiplayer, Co-op, Controller from categories
df['has_multiplayer'] = df['categories'].astype(str).str.contains('Multi-player|Online PvP|Online Multi-Player', case=False, na=False).astype(int)
df['has_coop'] = df['categories'].astype(str).str.contains('Co-op|Online Co-Op', case=False, na=False).astype(int)
df['has_controller'] = df['categories'].astype(str).str.contains('controller|gamepad|Full controller', case=False, na=False).astype(int)
df['has_trading_cards'] = df['categories'].astype(str).str.contains('Steam Trading Cards', case=False, na=False).astype(int)
df['has_workshop'] = df['categories'].astype(str).str.contains('Steam Workshop', case=False, na=False).astype(int)

# Store page quality signals
df['has_website'] = df['website'].notna().astype(int)
df = df.drop(columns=['website'], errors='ignore')

# Platform support (already boolean columns)
df['supports_mac'] = df['mac'].astype(int)
df['supports_linux'] = df['linux'].astype(int)

# Drop the raw boolean platform columns
df = df.drop(columns=['windows', 'mac', 'linux'], errors='ignore')

binary_features = ['is_free', 'has_achievements', 'has_dlc', 'is_early_access',
                   'has_multiplayer', 'has_coop', 'has_controller', 'has_trading_cards',
                   'has_workshop', 'has_website', 'supports_mac', 'supports_linux']

for feat in binary_features:
    ones = df[feat].sum()
    pct = ones / len(df) * 100
    log(f"  {feat}: {ones:,} games = {pct:.1f}%")

# ═══════════════════════════════════════════════════════════════
#  STEP 6: LANGUAGE FLAGS (8 STRATEGIC MARKETS)
# ═══════════════════════════════════════════════════════════════
log("\n--- STEP 6: Creating 8 Language Flags ---")

target_languages = {
    'lang_chinese': 'Chinese',
    'lang_japanese': 'Japanese',
    'lang_german': 'German',
    'lang_french': 'French',
    'lang_russian': 'Russian',
    'lang_korean': 'Korean',
    'lang_brazilian_pt': 'Portuguese - Brazil',
    'lang_spanish': 'Spanish',
}

for col_name, lang in target_languages.items():
    df[col_name] = df['supported_languages'].astype(str).str.contains(lang, case=False, na=False).astype(int)
    count = df[col_name].sum()
    log(f"  {col_name}: {count:,} games ({count/len(df)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
#  STEP 7: GENRE ONE-HOT ENCODING (10 GENRES)
# ═══════════════════════════════════════════════════════════════
log("\n--- STEP 7: Genre One-Hot Encoding (10 Genres) ---")

def parse_list(val):
    """Parse string representations of lists."""
    if pd.isna(val) or val == '':
        return []
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else []
    except:
        return []

df['_genre_list'] = df['genres'].apply(parse_list)

top_genres = ['Indie', 'Action', 'Adventure', 'RPG', 'Strategy',
              'Simulation', 'Casual', 'Sports', 'Racing', 'Free To Play']

for g in top_genres:
    col = f'genre_{g.lower().replace(" ", "_")}'
    df[col] = df['_genre_list'].apply(lambda x: 1 if g in x else 0)
    count = df[col].sum()
    log(f"  {col}: {count:,} games ({count/len(df)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
#  STEP 8: TAG ONE-HOT ENCODING (TOP 25 TAGS BY SIGNAL)
# ═══════════════════════════════════════════════════════════════
log("\n--- STEP 8: Tag One-Hot Encoding (Top 25 Tags) + Tag Count ---")

def parse_tag_dict(val):
    """Parse tag dictionary string and return list of tag names."""
    if pd.isna(val) or val == '':
        return []
    try:
        val_fixed = val.replace("'", '"')
        d = json.loads(val_fixed)
        return list(d.keys()) if isinstance(d, dict) else []
    except:
        return []

df['_tag_list'] = df['tags'].apply(parse_tag_dict)

# Tag count (store discoverability signal)
df['tag_count'] = df['_tag_list'].apply(len)
log(f"  tag_count: mean={df['tag_count'].mean():.1f}, median={df['tag_count'].median():.0f}")

# Top 25 tags ranked by signal strength (Flop vs Commercial Hit difference)
# from feasibility_report.txt
top_25_tags = [
    'Multiplayer',        # +36.5%
    'Singleplayer',       # +31.6%
    'Action',             # +26.3%
    'Co-op',              # +21.5%
    'Adventure',          # +21.3%
    'Great Soundtrack',   # +21.3%
    'Open World',         # +20.2%
    'Atmospheric',        # +18.7%
    'RPG',                # +18.4%
    'Story Rich',         # +17.0%
    'Strategy',           # +16.5%
    'Free to Play',       # +13.7%
    'Online Co-Op',       # +12.7%
    'First-Person',       # +12.4%
    'Shooter',            # +12.1%
    'Sandbox',            # +11.8%
    'Third Person',       # +11.0%
    'Simulation',         # +11.0%
    'Funny',              # +10.9%
    'Difficult',          # +10.9%
    # Negative signal tags (more common in Flops)
    'Casual',             # -6.0%
    '2D Platformer',      # -4.1%
    'Linear',             # -3.7%
    'Hidden Object',      # -2.7%
    'Minimalist',         # -2.6%
]

for t in top_25_tags:
    col = f'tag_{t.lower().replace(" ", "_").replace("-", "_")}'
    df[col] = df['_tag_list'].apply(lambda x: 1 if t in x else 0)
    count = df[col].sum()
    log(f"  {col}: {count:,} ({count/len(df)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
#  STEP 9: STORE PAGE QUALITY FEATURES
# ═══════════════════════════════════════════════════════════════
log("\n--- STEP 9: Store Page Quality Features ---")

# Screenshot count (store page richness indicator)
def count_list_safe(val):
    if pd.isna(val) or val == '' or val == '[]': return 0
    try: return len(ast.literal_eval(val))
    except: return 0

df['screenshot_count'] = df['screenshots'].apply(count_list_safe)
log(f"  screenshot_count: mean={df['screenshot_count'].mean():.1f}, median={df['screenshot_count'].median():.0f}")

# Drop raw media columns now that we've extracted features
df = df.drop(columns=['screenshots', 'movies'], errors='ignore')

# Review metrics kept for reference (excluded at model training time)
log(f"  pct_pos_total: mean={df['pct_pos_total'].mean():.1f}, median={df['pct_pos_total'].median()}")
log(f"  num_reviews_total: mean={df['num_reviews_total'].mean():.1f}, median={df['num_reviews_total'].median()}")
log(f"  peak_ccu: mean={df['peak_ccu'].mean():.1f}, median={df['peak_ccu'].median()}")

# ═══════════════════════════════════════════════════════════════
#  FINAL CLEANUP & EXPORT
# ═══════════════════════════════════════════════════════════════
log("\n--- FINAL: Cleanup & Export ---")

# Drop all remaining raw text/categorical columns
final_drops = ['genres', 'tags', 'categories', 'supported_languages',
               'full_audio_languages', 'estimated_owners', 'owners_lower',
               'positive', 'negative',  # redundant with pct_pos_total
               '_genre_list', '_tag_list']
df = df.drop(columns=[c for c in final_drops if c in df.columns], errors='ignore')

# Fill any remaining NaN with 0
nan_count = df.isna().sum().sum()
log(f"  Remaining NaN values: {nan_count}")
df = df.fillna(0)

# Export final dataset
df.to_csv(OUTPUT_FILE, index=False)
log(f"\n  Exported: {OUTPUT_FILE}")
log(f"  Final dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
log(f"  Columns: {list(df.columns)}")

# ═══════════════════════════════════════════════════════════════
#  WRITE REPORT
# ═══════════════════════════════════════════════════════════════
log(f"\n{'=' * 70}")
log(f"  Pipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"{'=' * 70}")

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\nReport saved to: {REPORT_FILE}")
