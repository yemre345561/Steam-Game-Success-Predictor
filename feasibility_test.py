"""
=============================================================================
FEASIBILITY ANALYSIS: Can Our Features Predict Success Tiers?
=============================================================================
Purpose: Before committing to any model or thesis chapter, we test whether
         there is ACTUAL SIGNAL in the features we plan to use.
         
Question: Do successful and unsuccessful games look DIFFERENT 
          when we compare their price, tags, languages, etc.?

If YES → The model has something to learn → We proceed confidently
If NO  → The features are useless → We must rethink our approach
=============================================================================
"""

import pandas as pd
import numpy as np
import ast
import collections
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD & PREPARE
# ============================================================================
print("Loading dataset...")
df = pd.read_csv("games_march2025_cleaned.csv")
print(f"Loaded: {len(df):,} games\n")

# --- Parse estimated_owners to get min_owners ---
def parse_min_owners(value):
    try:
        return int(str(value).replace(',', '').split(' - ')[0].strip())
    except:
        return 0

df['min_owners'] = df['estimated_owners'].apply(parse_min_owners)

# --- Define Success Tiers (Owners-based only) ---
def assign_tier(row):
    owners = row['min_owners']
    if owners < 20000:
        return 0  # Flop
    elif owners < 200000:
        return 1  # Niche Hit
    elif owners < 5000000:
        return 2  # Commercial Hit
    else:
        return 3  # Mega Hit

df['success_tier'] = df.apply(assign_tier, axis=1)

tier_names = {0: 'Flop', 1: 'Niche Hit', 2: 'Commercial Hit', 3: 'Mega Hit'}
df['tier_name'] = df['success_tier'].map(tier_names)

# ============================================================================
# REPORT
# ============================================================================
with open("feasibility_report.txt", "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("  FEASIBILITY ANALYSIS: Can Our Features Predict Success?\n")
    f.write("=" * 80 + "\n\n")

    # --- TIER DISTRIBUTION ---
    f.write("TIER DISTRIBUTION\n")
    f.write("-" * 50 + "\n")
    for tier in [0, 1, 2, 3]:
        count = (df['success_tier'] == tier).sum()
        pct = count / len(df) * 100
        f.write(f"  {tier_names[tier]:<20} {count:>8,} ({pct:.2f}%)\n")
    f.write(f"\n  Total: {len(df):,}\n\n")

    # ====================================================================
    # TEST 1: PRICE — Does price differ across tiers?
    # ====================================================================
    f.write("\n" + "=" * 80 + "\n")
    f.write("  TEST 1: PRICE vs SUCCESS TIER\n")
    f.write("  Question: Do successful games have different pricing?\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"{'Tier':<20} {'Median':>10} {'Mean':>10} {'% Free':>10} {'% > $20':>10}\n")
    f.write("-" * 65 + "\n")
    for tier in [0, 1, 2, 3]:
        subset = df[df['success_tier'] == tier]
        median_p = subset['price'].median()
        mean_p = subset['price'].mean()
        pct_free = (subset['price'] == 0).sum() / len(subset) * 100
        pct_over20 = (subset['price'] > 20).sum() / len(subset) * 100
        f.write(f"  {tier_names[tier]:<20} ${median_p:>8.2f} ${mean_p:>8.2f} {pct_free:>9.1f}% {pct_over20:>9.1f}%\n")

    # VERDICT
    flop_median = df[df['success_tier'] == 0]['price'].median()
    hit_median = df[df['success_tier'] == 2]['price'].median()
    if abs(hit_median - flop_median) > 2:
        f.write("\n  >>> SIGNAL DETECTED: Price differs significantly across tiers.\n")
    else:
        f.write("\n  >>> WEAK SIGNAL: Price alone may not be a strong predictor.\n")

    # ====================================================================
    # TEST 2: LANGUAGE COUNT — Does localization matter?
    # ====================================================================
    f.write("\n" + "=" * 80 + "\n")
    f.write("  TEST 2: LANGUAGE SUPPORT vs SUCCESS TIER\n")
    f.write("  Question: Do successful games support more languages?\n")
    f.write("=" * 80 + "\n\n")

    def safe_parse_list(val):
        if pd.isna(val) or val == '' or val == '[]':
            return []
        try:
            return ast.literal_eval(val)
        except:
            return []

    df['lang_count'] = df['supported_languages'].apply(safe_parse_list).apply(len)

    f.write(f"{'Tier':<20} {'Median Langs':>15} {'Mean Langs':>15} {'% with Chinese':>15}\n")
    f.write("-" * 70 + "\n")
    for tier in [0, 1, 2, 3]:
        subset = df[df['success_tier'] == tier]
        median_l = subset['lang_count'].median()
        mean_l = subset['lang_count'].mean()
        # Check Chinese support
        has_chinese = subset['supported_languages'].apply(
            lambda x: 'Simplified Chinese' in str(x)).sum() / len(subset) * 100
        f.write(f"  {tier_names[tier]:<20} {median_l:>14.0f} {mean_l:>14.1f} {has_chinese:>14.1f}%\n")

    flop_lang = df[df['success_tier'] == 0]['lang_count'].mean()
    hit_lang = df[df['success_tier'] == 2]['lang_count'].mean()
    f.write(f"\n  >>> Flop avg languages: {flop_lang:.1f} vs Commercial Hit avg: {hit_lang:.1f}")
    if hit_lang > flop_lang * 2:
        f.write("\n  >>> STRONG SIGNAL: Language support is a major differentiator!\n")
    else:
        f.write("\n  >>> SIGNAL DETECTED: Language count differs across tiers.\n")

    # ====================================================================
    # TEST 3: PLATFORM SUPPORT — Does multi-platform matter?
    # ====================================================================
    f.write("\n" + "=" * 80 + "\n")
    f.write("  TEST 3: PLATFORM SUPPORT vs SUCCESS TIER\n")
    f.write("  Question: Do successful games support Mac/Linux more?\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"{'Tier':<20} {'% Windows':>12} {'% Mac':>10} {'% Linux':>10}\n")
    f.write("-" * 55 + "\n")
    for tier in [0, 1, 2, 3]:
        subset = df[df['success_tier'] == tier]
        pct_win = subset['windows'].sum() / len(subset) * 100
        pct_mac = subset['mac'].sum() / len(subset) * 100
        pct_linux = subset['linux'].sum() / len(subset) * 100
        f.write(f"  {tier_names[tier]:<20} {pct_win:>11.1f}% {pct_mac:>9.1f}% {pct_linux:>9.1f}%\n")

    # ====================================================================
    # TEST 4: ACHIEVEMENTS — Does having achievements matter?
    # ====================================================================
    f.write("\n" + "=" * 80 + "\n")
    f.write("  TEST 4: ACHIEVEMENTS vs SUCCESS TIER\n")
    f.write("  Question: Do successful games have more achievements?\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"{'Tier':<20} {'% Has Achiev.':>15} {'Median Count':>15} {'Mean Count':>12}\n")
    f.write("-" * 65 + "\n")
    for tier in [0, 1, 2, 3]:
        subset = df[df['success_tier'] == tier]
        has_ach = (subset['achievements'] > 0).sum() / len(subset) * 100
        median_a = subset[subset['achievements'] > 0]['achievements'].median() if (subset['achievements'] > 0).any() else 0
        mean_a = subset['achievements'].mean()
        f.write(f"  {tier_names[tier]:<20} {has_ach:>14.1f}% {median_a:>14.0f} {mean_a:>11.1f}\n")

    # ====================================================================
    # TEST 5: DLC COUNT — Does DLC strategy matter?
    # ====================================================================
    f.write("\n" + "=" * 80 + "\n")
    f.write("  TEST 5: DLC COUNT vs SUCCESS TIER\n")
    f.write("  Question: Do successful games offer more DLCs?\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"{'Tier':<20} {'% Has DLC':>12} {'Mean DLCs':>12}\n")
    f.write("-" * 48 + "\n")
    for tier in [0, 1, 2, 3]:
        subset = df[df['success_tier'] == tier]
        has_dlc = (subset['dlc_count'] > 0).sum() / len(subset) * 100
        mean_dlc = subset['dlc_count'].mean()
        f.write(f"  {tier_names[tier]:<20} {has_dlc:>11.1f}% {mean_dlc:>11.1f}\n")

    # ====================================================================
    # TEST 6: EARLY ACCESS — Does EA help or hurt?
    # ====================================================================
    f.write("\n" + "=" * 80 + "\n")
    f.write("  TEST 6: EARLY ACCESS vs SUCCESS TIER\n")
    f.write("  Question: Is Early Access more common in successful games?\n")
    f.write("=" * 80 + "\n\n")

    df['is_early_access'] = df['genres'].apply(lambda x: 'Early Access' in str(x))

    f.write(f"{'Tier':<20} {'% Early Access':>15}\n")
    f.write("-" * 38 + "\n")
    for tier in [0, 1, 2, 3]:
        subset = df[df['success_tier'] == tier]
        ea_pct = subset['is_early_access'].sum() / len(subset) * 100
        f.write(f"  {tier_names[tier]:<20} {ea_pct:>14.1f}%\n")

    # ====================================================================
    # TEST 7: CATEGORIES (Multiplayer/Co-op) — Social features matter?
    # ====================================================================
    f.write("\n" + "=" * 80 + "\n")
    f.write("  TEST 7: MULTIPLAYER & CO-OP vs SUCCESS TIER\n")
    f.write("  Question: Do social features correlate with success?\n")
    f.write("=" * 80 + "\n\n")

    df['has_multiplayer'] = df['categories'].apply(lambda x: 'Multi-player' in str(x))
    df['has_coop'] = df['categories'].apply(lambda x: 'Co-op' in str(x))

    f.write(f"{'Tier':<20} {'% Multiplayer':>15} {'% Co-op':>10}\n")
    f.write("-" * 48 + "\n")
    for tier in [0, 1, 2, 3]:
        subset = df[df['success_tier'] == tier]
        mp = subset['has_multiplayer'].sum() / len(subset) * 100
        coop = subset['has_coop'].sum() / len(subset) * 100
        f.write(f"  {tier_names[tier]:<20} {mp:>14.1f}% {coop:>9.1f}%\n")

    # ====================================================================
    # TEST 8: TOP TAGS — Which tags appear more in successful games?
    # ====================================================================
    f.write("\n" + "=" * 80 + "\n")
    f.write("  TEST 8: TAG ANALYSIS — Success Tags vs Flop Tags\n")
    f.write("  Question: Which tags are overrepresented in hits vs flops?\n")
    f.write("=" * 80 + "\n\n")

    def safe_parse_dict(val):
        if pd.isna(val) or val == '' or val == '{}':
            return {}
        try:
            return ast.literal_eval(val)
        except:
            return {}

    # Get tag presence per tier
    tier_tag_rates = {}
    for tier in [0, 2, 3]:  # Compare Flop vs Commercial vs Mega
        subset = df[df['success_tier'] == tier]
        tag_counts = collections.Counter()
        for tags in subset['tags'].apply(safe_parse_dict):
            for tag in tags:
                tag_counts[tag] += 1
        # Convert to percentage
        tier_tag_rates[tier] = {tag: count / len(subset) * 100 
                                 for tag, count in tag_counts.items()}

    # Find tags that are MUCH more common in Commercial/Mega hits than in Flops
    f.write("Tags that are MORE common in COMMERCIAL HITS than FLOPS:\n")
    f.write(f"{'Tag':<30} {'In Flops':>10} {'In Comm.Hit':>12} {'Difference':>12}\n")
    f.write("-" * 68 + "\n")

    success_tags = []
    for tag in tier_tag_rates.get(2, {}):
        flop_rate = tier_tag_rates.get(0, {}).get(tag, 0)
        hit_rate = tier_tag_rates.get(2, {}).get(tag, 0)
        if flop_rate > 1 and hit_rate > 1:  # Must appear in both
            diff = hit_rate - flop_rate
            success_tags.append((tag, flop_rate, hit_rate, diff))

    success_tags.sort(key=lambda x: x[3], reverse=True)
    for tag, flop_r, hit_r, diff in success_tags[:20]:
        f.write(f"  {tag:<30} {flop_r:>9.1f}% {hit_r:>11.1f}% {diff:>+11.1f}%\n")

    f.write("\n\nTags that are MORE common in FLOPS than COMMERCIAL HITS:\n")
    f.write(f"{'Tag':<30} {'In Flops':>10} {'In Comm.Hit':>12} {'Difference':>12}\n")
    f.write("-" * 68 + "\n")

    flop_tags = sorted(success_tags, key=lambda x: x[3])
    for tag, flop_r, hit_r, diff in flop_tags[:20]:
        f.write(f"  {tag:<30} {flop_r:>9.1f}% {hit_r:>11.1f}% {diff:>+11.1f}%\n")

    # ====================================================================
    # TEST 9: RELEASE YEAR — Does era matter?
    # ====================================================================
    f.write("\n" + "=" * 80 + "\n")
    f.write("  TEST 9: RELEASE YEAR vs SUCCESS TIER\n")
    f.write("  Question: Are older games more likely to be hits (survivorship)?\n")
    f.write("=" * 80 + "\n\n")

    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    f.write(f"{'Tier':<20} {'Median Year':>12} {'Mean Year':>12}\n")
    f.write("-" * 48 + "\n")
    for tier in [0, 1, 2, 3]:
        subset = df[df['success_tier'] == tier]
        valid = subset[subset['release_year'].notna()]
        if len(valid) > 0:
            f.write(f"  {tier_names[tier]:<20} {valid['release_year'].median():>11.0f} {valid['release_year'].mean():>11.1f}\n")

    # ====================================================================
    # TEST 10: GENRE COMBINATIONS — What genres succeed?
    # ====================================================================
    f.write("\n" + "=" * 80 + "\n")
    f.write("  TEST 10: GENRE DISTRIBUTION ACROSS TIERS\n")
    f.write("=" * 80 + "\n\n")

    key_genres = ['Indie', 'Action', 'Adventure', 'RPG', 'Strategy', 
                  'Simulation', 'Casual', 'Sports', 'Racing', 'Free To Play']

    header = f"{'Genre':<20}"
    for tier in [0, 1, 2, 3]:
        header += f" {tier_names[tier]:>15}"
    f.write(header + "\n" + "-" * 80 + "\n")

    for genre in key_genres:
        row = f"  {genre:<20}"
        for tier in [0, 1, 2, 3]:
            subset = df[df['success_tier'] == tier]
            pct = subset['genres'].apply(lambda x: genre in str(x)).sum() / len(subset) * 100
            row += f" {pct:>14.1f}%"
        f.write(row + "\n")

    # ====================================================================
    # FINAL VERDICT
    # ====================================================================
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("  FINAL VERDICT: IS THERE ENOUGH SIGNAL?\n")
    f.write("=" * 80 + "\n\n")

    f.write("""
FEATURE-BY-FEATURE ASSESSMENT:

  Feature               Signal Strength    Verdict
  -------               ---------------    -------
  Price                  See results above  [CHECK REPORT]
  Language Count         See results above  [CHECK REPORT]
  Platform Support       See results above  [CHECK REPORT]
  Achievements           See results above  [CHECK REPORT]
  DLC Count              See results above  [CHECK REPORT]
  Early Access           See results above  [CHECK REPORT]
  Multiplayer/Co-op      See results above  [CHECK REPORT]
  Tags                   See results above  [CHECK REPORT]
  Release Year           See results above  [CHECK REPORT]
  Genre                  See results above  [CHECK REPORT]

OVERALL: If the numbers above show CLEAR DIFFERENCES between tiers,
then there IS signal for the model to learn. If numbers are similar
across all tiers, the feature is useless for prediction.
""")

print("=" * 60)
print("  FEASIBILITY REPORT COMPLETE!")
print("  Output: feasibility_report.txt")
print("=" * 60)
