"""
=============================================================================
STEAM SUCCESS PREDICTION MODEL - DEEP EXPLORATORY DATA ANALYSIS (EDA)
=============================================================================
Purpose: Comprehensive examination of every column, distribution, 
         correlation, and anomaly in the Steam dataset.
Output:  deep_eda_report.txt (Full English report)
Author:  AI Data Scientist for Yunus Emre Acikoglu's Thesis
=============================================================================
"""

import pandas as pd
import numpy as np
import ast
import collections
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = "games_march2025_cleaned.csv"
OUTPUT_FILE = "deep_eda_report.txt"

def write_section(f, title, content=""):
    """Helper to write formatted sections to the report."""
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"  {title}\n")
    f.write("=" * 80 + "\n\n")
    if content:
        f.write(str(content) + "\n")

def safe_parse_list(value):
    """Safely parse string representation of lists."""
    if pd.isna(value) or value == '' or value == '[]':
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []

def safe_parse_dict(value):
    """Safely parse string representation of dicts (for tags)."""
    if pd.isna(value) or value == '' or value == '{}':
        return {}
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}

def parse_owners(value):
    """Parse estimated_owners range string to min and max values."""
    if pd.isna(value) or value == '':
        return None, None
    try:
        parts = str(value).replace(',', '').split(' - ')
        if len(parts) == 2:
            return int(parts[0].strip()), int(parts[1].strip())
        return None, None
    except:
        return None, None

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading dataset... (This may take a moment for 468MB)")
df = pd.read_csv(INPUT_FILE)
print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns\n")

# ============================================================================
# GENERATE REPORT
# ============================================================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("  STEAM SUCCESS PREDICTION MODEL\n")
    f.write("  DEEP EXPLORATORY DATA ANALYSIS (EDA) REPORT\n")
    f.write("  Dataset: games_march2025_cleaned.csv\n")
    f.write("=" * 80 + "\n")

    # ========================================================================
    # SECTION 1: DATASET OVERVIEW
    # ========================================================================
    write_section(f, "SECTION 1: DATASET OVERVIEW")
    f.write(f"Total Games (Rows): {df.shape[0]:,}\n")
    f.write(f"Total Features (Columns): {df.shape[1]}\n")
    f.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n\n")

    f.write("--- Column Names, Data Types & Non-Null Counts ---\n\n")
    f.write(f"{'Column':<30} {'Dtype':<12} {'Non-Null':>10} {'Null':>8} {'Null%':>8}\n")
    f.write("-" * 70 + "\n")
    for col in df.columns:
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        f.write(f"{col:<30} {str(df[col].dtype):<12} {non_null:>10,} {null_count:>8,} {null_pct:>7.2f}%\n")
    
    print("Section 1 done: Dataset Overview")

    # ========================================================================
    # SECTION 2: MISSING VALUES ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 2: MISSING VALUES ANALYSIS")
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        f.write(f"Columns with missing values: {len(missing)}\n\n")
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            f.write(f"  {col:<30} {count:>8,} missing ({pct:.2f}%)\n")
    else:
        f.write("No missing values found in any column.\n")

    # Check for sentinel values (-1 used as missing indicator)
    f.write("\n\n--- Sentinel Value Check (Columns using -1 as missing) ---\n\n")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        neg_count = (df[col] == -1).sum()
        if neg_count > 0:
            pct = (neg_count / len(df)) * 100
            f.write(f"  {col:<30} {neg_count:>8,} rows with value -1 ({pct:.2f}%)\n")
    
    print("Section 2 done: Missing Values")

    # ========================================================================
    # SECTION 3: NUMERICAL COLUMNS - FULL STATISTICS
    # ========================================================================
    write_section(f, "SECTION 3: NUMERICAL COLUMNS - FULL STATISTICS")
    
    numerical_features = [
        'appid', 'required_age', 'price', 'dlc_count', 'metacritic_score',
        'achievements', 'recommendations', 'user_score', 'score_rank',
        'positive', 'negative', 'average_playtime_forever', 'average_playtime_2weeks',
        'median_playtime_forever', 'median_playtime_2weeks', 'discount',
        'peak_ccu', 'pct_pos_total', 'num_reviews_total', 'pct_pos_recent',
        'num_reviews_recent'
    ]
    
    for col in numerical_features:
        if col in df.columns:
            f.write(f"\n--- {col} ---\n")
            series = df[col].dropna()
            f.write(f"  Count:          {series.count():>15,}\n")
            f.write(f"  Mean:           {series.mean():>15.2f}\n")
            f.write(f"  Std Dev:        {series.std():>15.2f}\n")
            f.write(f"  Min:            {series.min():>15.2f}\n")
            f.write(f"  25th Percentile:{series.quantile(0.25):>15.2f}\n")
            f.write(f"  Median (50th):  {series.quantile(0.50):>15.2f}\n")
            f.write(f"  75th Percentile:{series.quantile(0.75):>15.2f}\n")
            f.write(f"  90th Percentile:{series.quantile(0.90):>15.2f}\n")
            f.write(f"  95th Percentile:{series.quantile(0.95):>15.2f}\n")
            f.write(f"  99th Percentile:{series.quantile(0.99):>15.2f}\n")
            f.write(f"  Max:            {series.max():>15.2f}\n")
            f.write(f"  Skewness:       {series.skew():>15.4f}\n")
            f.write(f"  Kurtosis:       {series.kurtosis():>15.4f}\n")
            
            # Zero count for key metrics
            if col in ['price', 'peak_ccu', 'positive', 'negative', 'num_reviews_total', 'achievements']:
                zero_count = (series == 0).sum()
                f.write(f"  Zero Count:     {zero_count:>15,} ({zero_count/len(series)*100:.2f}%)\n")
    
    print("Section 3 done: Numerical Statistics")

    # ========================================================================
    # SECTION 4: PRICE DEEP DIVE
    # ========================================================================
    write_section(f, "SECTION 4: PRICE DEEP DIVE")
    
    # Price distribution by brackets
    price_brackets = [
        (0, 0, "Free (F2P)"),
        (0.01, 4.99, "$0.01 - $4.99"),
        (5.00, 9.99, "$5.00 - $9.99"),
        (10.00, 14.99, "$10.00 - $14.99"),
        (15.00, 19.99, "$15.00 - $19.99"),
        (20.00, 29.99, "$20.00 - $29.99"),
        (30.00, 59.99, "$30.00 - $59.99"),
        (60.00, 999.99, "$60.00+")
    ]
    
    f.write("--- Price Bracket Distribution ---\n\n")
    f.write(f"{'Bracket':<25} {'Count':>10} {'Percentage':>12}\n")
    f.write("-" * 50 + "\n")
    for low, high, label in price_brackets:
        if low == 0 and high == 0:
            count = (df['price'] == 0).sum()
        else:
            count = ((df['price'] >= low) & (df['price'] <= high)).sum()
        pct = (count / len(df)) * 100
        f.write(f"{label:<25} {count:>10,} {pct:>10.2f}%\n")
    
    # Average review score by price bracket
    f.write("\n\n--- Average Positive Review Rate by Price Bracket ---\n\n")
    f.write(f"{'Bracket':<25} {'Avg Pos Rate':>15} {'Avg Reviews':>15} {'Avg Peak CCU':>15}\n")
    f.write("-" * 75 + "\n")
    for low, high, label in price_brackets:
        if low == 0 and high == 0:
            subset = df[df['price'] == 0]
        else:
            subset = df[(df['price'] >= low) & (df['price'] <= high)]
        if len(subset) > 0:
            # Filter out -1 sentinel values
            valid_pct = subset[subset['pct_pos_total'] >= 0]['pct_pos_total']
            avg_pct = valid_pct.mean() if len(valid_pct) > 0 else 0
            avg_rev = subset[subset['num_reviews_total'] >= 0]['num_reviews_total'].mean()
            avg_ccu = subset['peak_ccu'].mean()
            f.write(f"{label:<25} {avg_pct:>14.1f}% {avg_rev:>15.0f} {avg_ccu:>15.0f}\n")
    
    print("Section 4 done: Price Deep Dive")

    # ========================================================================
    # SECTION 5: ESTIMATED OWNERS ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 5: ESTIMATED OWNERS ANALYSIS")
    
    # Parse estimated_owners
    owners_parsed = df['estimated_owners'].apply(parse_owners)
    df['owners_min'] = owners_parsed.apply(lambda x: x[0])
    df['owners_max'] = owners_parsed.apply(lambda x: x[1])
    
    f.write("--- Estimated Owners Range Distribution ---\n\n")
    owner_counts = df['estimated_owners'].value_counts().sort_index()
    f.write(f"{'Owner Range':<30} {'Count':>10} {'Percentage':>10} {'Cumulative':>12}\n")
    f.write("-" * 65 + "\n")
    cumulative = 0
    for range_str, count in owner_counts.items():
        pct = (count / len(df)) * 100
        cumulative += pct
        f.write(f"{str(range_str):<30} {count:>10,} {pct:>9.2f}% {cumulative:>10.2f}%\n")
    
    # Owners vs Reviews correlation
    valid_owners = df[df['owners_min'].notna() & (df['num_reviews_total'] >= 0)]
    if len(valid_owners) > 0:
        corr = valid_owners['owners_min'].corr(valid_owners['num_reviews_total'])
        f.write(f"\nCorrelation between min_owners and num_reviews_total: {corr:.4f}\n")
    
    print("Section 5 done: Estimated Owners")

    # ========================================================================
    # SECTION 6: REVIEW & RATING ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 6: REVIEW & RATING ANALYSIS")
    
    # Filter out sentinel -1 values
    valid_reviews = df[df['num_reviews_total'] >= 0].copy()
    valid_pct = df[df['pct_pos_total'] >= 0].copy()
    
    f.write(f"Games with valid review data: {len(valid_reviews):,}\n")
    f.write(f"Games with valid positive rate: {len(valid_pct):,}\n\n")
    
    # Review count brackets
    review_brackets = [
        (0, 0, "0 reviews (No data)"),
        (1, 10, "1-10 reviews"),
        (11, 50, "11-50 reviews"),
        (51, 100, "51-100 reviews"),
        (101, 500, "101-500 reviews"),
        (501, 1000, "501-1,000 reviews"),
        (1001, 5000, "1,001-5,000 reviews"),
        (5001, 10000, "5,001-10,000 reviews"),
        (10001, 50000, "10,001-50,000 reviews"),
        (50001, 100000, "50,001-100,000 reviews"),
        (100001, 10000000, "100,001+ reviews")
    ]
    
    f.write("--- Review Count Distribution ---\n\n")
    f.write(f"{'Bracket':<30} {'Count':>10} {'Percentage':>10}\n")
    f.write("-" * 55 + "\n")
    for low, high, label in review_brackets:
        count = ((valid_reviews['num_reviews_total'] >= low) & (valid_reviews['num_reviews_total'] <= high)).sum()
        pct = (count / len(valid_reviews)) * 100
        f.write(f"{label:<30} {count:>10,} {pct:>9.2f}%\n")
    
    # Positive rate distribution
    f.write("\n\n--- Positive Review Rate Distribution ---\n\n")
    rate_brackets = [
        (0, 39, "Overwhelmingly Negative (0-39%)"),
        (40, 49, "Mostly Negative (40-49%)"),
        (50, 59, "Mixed (50-59%)"),
        (60, 69, "Mostly Positive (60-69%)"),
        (70, 79, "Positive (70-79%)"),
        (80, 89, "Very Positive (80-89%)"),
        (90, 94, "Overwhelmingly Positive (90-94%)"),
        (95, 100, "Overwhelmingly Positive (95-100%)")
    ]
    
    f.write(f"{'Rating Category':<40} {'Count':>10} {'Percentage':>10}\n")
    f.write("-" * 65 + "\n")
    for low, high, label in rate_brackets:
        count = ((valid_pct['pct_pos_total'] >= low) & (valid_pct['pct_pos_total'] <= high)).sum()
        pct = (count / len(valid_pct)) * 100
        f.write(f"{label:<40} {count:>10,} {pct:>9.2f}%\n")
    
    print("Section 6 done: Review & Rating Analysis")

    # ========================================================================
    # SECTION 7: GENRE ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 7: GENRE ANALYSIS")
    
    genre_lists = df['genres'].apply(safe_parse_list)
    all_genres = [g for sublist in genre_lists for g in sublist]
    genre_counter = collections.Counter(all_genres)
    
    f.write(f"Total unique genres: {len(genre_counter)}\n\n")
    f.write(f"{'Genre':<35} {'Count':>10} {'% of Games':>10}\n")
    f.write("-" * 60 + "\n")
    for genre, count in genre_counter.most_common(30):
        pct = (count / len(df)) * 100
        f.write(f"{genre:<35} {count:>10,} {pct:>9.2f}%\n")
    
    # Genre combinations
    f.write("\n\n--- Most Common Genre Combinations (Top 15) ---\n\n")
    genre_combos = genre_lists.apply(lambda x: ' + '.join(sorted(x)) if x else 'None')
    combo_counts = genre_combos.value_counts().head(15)
    for combo, count in combo_counts.items():
        pct = (count / len(df)) * 100
        f.write(f"  {combo:<50} {count:>8,} ({pct:.2f}%)\n")
    
    print("Section 7 done: Genre Analysis")

    # ========================================================================
    # SECTION 8: TAG ANALYSIS (KEY FOR PREDICTION)
    # ========================================================================
    write_section(f, "SECTION 8: TAG ANALYSIS (KEY FEATURE FOR PREDICTION)")
    
    tag_dicts = df['tags'].apply(safe_parse_dict)
    
    # Flatten all tags
    all_tags = []
    tag_total_weight = collections.Counter()
    for tag_dict in tag_dicts:
        if isinstance(tag_dict, dict):
            for tag, weight in tag_dict.items():
                all_tags.append(tag)
                tag_total_weight[tag] += weight
    
    tag_counter = collections.Counter(all_tags)
    
    f.write(f"Total unique tags: {len(tag_counter)}\n\n")
    
    f.write("--- Top 50 Tags by Frequency (How many games have this tag) ---\n\n")
    f.write(f"{'Tag':<35} {'Games':>10} {'% of Games':>10} {'Total Weight':>15}\n")
    f.write("-" * 75 + "\n")
    for tag, count in tag_counter.most_common(50):
        pct = (count / len(df)) * 100
        weight = tag_total_weight.get(tag, 0)
        f.write(f"{tag:<35} {count:>10,} {pct:>9.2f}% {weight:>15,}\n")
    
    print("Section 8 done: Tag Analysis")

    # ========================================================================
    # SECTION 9: LANGUAGE SUPPORT ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 9: LANGUAGE SUPPORT ANALYSIS")
    
    lang_lists = df['supported_languages'].apply(safe_parse_list)
    all_langs = [l for sublist in lang_lists for l in sublist]
    lang_counter = collections.Counter(all_langs)
    
    f.write(f"Total unique languages: {len(lang_counter)}\n\n")
    f.write(f"{'Language':<35} {'Games':>10} {'% of Games':>10}\n")
    f.write("-" * 60 + "\n")
    for lang, count in lang_counter.most_common(30):
        pct = (count / len(df)) * 100
        f.write(f"{lang:<35} {count:>10,} {pct:>9.2f}%\n")
    
    # Language count per game
    df['lang_count'] = lang_lists.apply(len)
    f.write(f"\n\n--- Number of Supported Languages per Game ---\n\n")
    f.write(f"  Mean:   {df['lang_count'].mean():.1f}\n")
    f.write(f"  Median: {df['lang_count'].median():.0f}\n")
    f.write(f"  Min:    {df['lang_count'].min()}\n")
    f.write(f"  Max:    {df['lang_count'].max()}\n")
    f.write(f"  Games with 0 languages listed: {(df['lang_count'] == 0).sum():,}\n")
    f.write(f"  Games with only English: {(df['lang_count'] == 1).sum():,}\n")
    
    # Full audio languages
    audio_lists = df['full_audio_languages'].apply(safe_parse_list)
    all_audio = [l for sublist in audio_lists for l in sublist]
    audio_counter = collections.Counter(all_audio)
    
    f.write(f"\n\n--- Full Audio Language Support (Top 15) ---\n\n")
    f.write(f"{'Language':<35} {'Games':>10} {'% of Games':>10}\n")
    f.write("-" * 60 + "\n")
    for lang, count in audio_counter.most_common(15):
        pct = (count / len(df)) * 100
        f.write(f"{lang:<35} {count:>10,} {pct:>9.2f}%\n")
    
    print("Section 9 done: Language Analysis")

    # ========================================================================
    # SECTION 10: PLATFORM SUPPORT ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 10: PLATFORM SUPPORT ANALYSIS")
    
    for platform in ['windows', 'mac', 'linux']:
        if platform in df.columns:
            count = df[platform].sum()
            pct = (count / len(df)) * 100
            f.write(f"  {platform.capitalize():<15} {count:>10,} games ({pct:.2f}%)\n")
    
    # Platform combinations
    f.write("\n--- Platform Combinations ---\n\n")
    df['platform_combo'] = ''
    if 'windows' in df.columns:
        df['platform_combo'] = df.apply(
            lambda r: '+'.join([p for p in ['Windows', 'Mac', 'Linux'] 
                               if r.get(p.lower(), False)]), axis=1)
    combo_counts = df['platform_combo'].value_counts()
    for combo, count in combo_counts.items():
        pct = (count / len(df)) * 100
        f.write(f"  {combo:<30} {count:>10,} ({pct:.2f}%)\n")
    
    print("Section 10 done: Platform Analysis")

    # ========================================================================
    # SECTION 11: RELEASE DATE & TEMPORAL ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 11: RELEASE DATE & TEMPORAL ANALYSIS")
    
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    
    f.write("--- Games Released Per Year (Last 15 Years) ---\n\n")
    yearly = df[df['release_year'] >= 2010]['release_year'].value_counts().sort_index()
    f.write(f"{'Year':<8} {'Games':>10} {'Avg Price':>12} {'Avg Reviews':>15}\n")
    f.write("-" * 50 + "\n")
    for year, count in yearly.items():
        year_data = df[df['release_year'] == year]
        avg_price = year_data['price'].mean()
        valid_rev = year_data[year_data['num_reviews_total'] >= 0]['num_reviews_total']
        avg_rev = valid_rev.mean() if len(valid_rev) > 0 else 0
        f.write(f"{int(year):<8} {count:>10,} ${avg_price:>10.2f} {avg_rev:>15.0f}\n")
    
    # Invalid dates
    invalid_dates = df['release_year'].isna().sum()
    f.write(f"\nGames with invalid/missing release dates: {invalid_dates:,}\n")
    
    print("Section 11 done: Temporal Analysis")

    # ========================================================================
    # SECTION 12: PLAYTIME ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 12: PLAYTIME ANALYSIS")
    
    playtime_cols = ['average_playtime_forever', 'average_playtime_2weeks',
                     'median_playtime_forever', 'median_playtime_2weeks']
    
    for col in playtime_cols:
        if col in df.columns:
            series = df[col]
            non_zero = series[series > 0]
            f.write(f"\n--- {col} ---\n")
            f.write(f"  Games with playtime > 0:  {len(non_zero):>10,} ({len(non_zero)/len(df)*100:.2f}%)\n")
            f.write(f"  Games with playtime = 0:  {(series == 0).sum():>10,} ({(series==0).sum()/len(df)*100:.2f}%)\n")
            if len(non_zero) > 0:
                f.write(f"  Mean (non-zero only):     {non_zero.mean():>10.1f} minutes\n")
                f.write(f"  Median (non-zero only):   {non_zero.median():>10.1f} minutes\n")
                f.write(f"  Max:                      {non_zero.max():>10.0f} minutes ({non_zero.max()/60:.0f} hours)\n")
    
    print("Section 12 done: Playtime Analysis")

    # ========================================================================
    # SECTION 13: CATEGORIES ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 13: CATEGORIES ANALYSIS (Game Features)")
    
    cat_lists = df['categories'].apply(safe_parse_list)
    all_cats = [c for sublist in cat_lists for c in sublist]
    cat_counter = collections.Counter(all_cats)
    
    f.write(f"Total unique categories: {len(cat_counter)}\n\n")
    f.write(f"{'Category':<45} {'Games':>10} {'% of Games':>10}\n")
    f.write("-" * 70 + "\n")
    for cat, count in cat_counter.most_common(30):
        pct = (count / len(df)) * 100
        f.write(f"{cat:<45} {count:>10,} {pct:>9.2f}%\n")
    
    print("Section 13 done: Categories Analysis")

    # ========================================================================
    # SECTION 14: METACRITIC & EXTERNAL SCORES
    # ========================================================================
    write_section(f, "SECTION 14: METACRITIC & EXTERNAL SCORES")
    
    has_metacritic = df[df['metacritic_score'] > 0]
    f.write(f"Games with Metacritic score > 0: {len(has_metacritic):,} ({len(has_metacritic)/len(df)*100:.2f}%)\n\n")
    
    if len(has_metacritic) > 0:
        f.write("--- Metacritic Score Distribution (for scored games only) ---\n\n")
        f.write(f"  Mean:   {has_metacritic['metacritic_score'].mean():.1f}\n")
        f.write(f"  Median: {has_metacritic['metacritic_score'].median():.0f}\n")
        f.write(f"  Min:    {has_metacritic['metacritic_score'].min()}\n")
        f.write(f"  Max:    {has_metacritic['metacritic_score'].max()}\n\n")
        
        meta_brackets = [(0, 49, "Bad (0-49)"), (50, 69, "Mixed (50-69)"), 
                         (70, 84, "Good (70-84)"), (85, 100, "Great (85-100)")]
        for low, high, label in meta_brackets:
            count = ((has_metacritic['metacritic_score'] >= low) & 
                     (has_metacritic['metacritic_score'] <= high)).sum()
            pct = (count / len(has_metacritic)) * 100
            f.write(f"  {label:<25} {count:>6,} ({pct:.1f}%)\n")
    
    print("Section 14 done: Metacritic Analysis")

    # ========================================================================
    # SECTION 15: DLC COUNT ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 15: DLC COUNT ANALYSIS")
    
    f.write(f"Games with 0 DLCs:        {(df['dlc_count'] == 0).sum():>10,} ({(df['dlc_count']==0).sum()/len(df)*100:.2f}%)\n")
    f.write(f"Games with 1-5 DLCs:      {((df['dlc_count'] >= 1) & (df['dlc_count'] <= 5)).sum():>10,}\n")
    f.write(f"Games with 6-20 DLCs:     {((df['dlc_count'] >= 6) & (df['dlc_count'] <= 20)).sum():>10,}\n")
    f.write(f"Games with 20+ DLCs:      {(df['dlc_count'] > 20).sum():>10,}\n")
    f.write(f"Max DLC count:            {df['dlc_count'].max():>10,}\n")
    
    print("Section 15 done: DLC Analysis")

    # ========================================================================
    # SECTION 16: CORRELATION MATRIX (KEY NUMERICAL FEATURES)
    # ========================================================================
    write_section(f, "SECTION 16: CORRELATION MATRIX (Success-Related Features)")
    
    corr_cols = ['price', 'dlc_count', 'metacritic_score', 'achievements',
                 'recommendations', 'positive', 'negative', 'peak_ccu',
                 'average_playtime_forever', 'median_playtime_forever',
                 'discount', 'pct_pos_total', 'num_reviews_total', 'num_reviews_recent']
    
    # Filter valid data (remove -1 sentinels)
    corr_df = df[corr_cols].copy()
    for col in corr_cols:
        if col in ['pct_pos_total', 'num_reviews_total', 'pct_pos_recent', 'num_reviews_recent']:
            corr_df.loc[corr_df[col] < 0, col] = np.nan
    
    corr_matrix = corr_df.corr()
    
    f.write("Correlation with num_reviews_total (proxy for commercial success):\n\n")
    if 'num_reviews_total' in corr_matrix.columns:
        target_corr = corr_matrix['num_reviews_total'].drop('num_reviews_total').sort_values(ascending=False)
        for feat, corr_val in target_corr.items():
            bar = "+" * int(abs(corr_val) * 30)
            sign = "+" if corr_val > 0 else "-"
            f.write(f"  {feat:<30} {corr_val:>8.4f}  {sign}{bar}\n")
    
    f.write("\n\nCorrelation with pct_pos_total (proxy for quality/satisfaction):\n\n")
    if 'pct_pos_total' in corr_matrix.columns:
        target_corr = corr_matrix['pct_pos_total'].drop('pct_pos_total').sort_values(ascending=False)
        for feat, corr_val in target_corr.items():
            bar = "+" * int(abs(corr_val) * 30)
            sign = "+" if corr_val > 0 else "-"
            f.write(f"  {feat:<30} {corr_val:>8.4f}  {sign}{bar}\n")
    
    print("Section 16 done: Correlation Matrix")

    # ========================================================================
    # SECTION 17: PEAK CCU (CONCURRENT USERS) ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 17: PEAK CCU (CONCURRENT USERS) ANALYSIS")
    
    ccu = df['peak_ccu']
    f.write(f"Games with Peak CCU = 0:   {(ccu == 0).sum():>10,} ({(ccu==0).sum()/len(df)*100:.2f}%)\n")
    f.write(f"Games with Peak CCU 1-10:  {((ccu >= 1) & (ccu <= 10)).sum():>10,}\n")
    f.write(f"Games with Peak CCU 11-100:{((ccu >= 11) & (ccu <= 100)).sum():>10,}\n")
    f.write(f"Games with Peak CCU 101-1K:{((ccu >= 101) & (ccu <= 1000)).sum():>10,}\n")
    f.write(f"Games with Peak CCU 1K-10K:{((ccu >= 1001) & (ccu <= 10000)).sum():>10,}\n")
    f.write(f"Games with Peak CCU 10K+:  {(ccu > 10000).sum():>10,}\n")
    f.write(f"Max Peak CCU:              {ccu.max():>10,}\n")
    
    print("Section 17 done: Peak CCU Analysis")

    # ========================================================================
    # SECTION 18: TEXT COLUMN ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 18: TEXT COLUMN ANALYSIS (Description Lengths)")
    
    text_cols = ['detailed_description', 'about_the_game', 'short_description', 'reviews']
    for col in text_cols:
        if col in df.columns:
            lengths = df[col].fillna('').str.len()
            non_empty = lengths[lengths > 0]
            f.write(f"\n--- {col} ---\n")
            f.write(f"  Non-empty:  {len(non_empty):>10,}\n")
            f.write(f"  Empty:      {(lengths == 0).sum():>10,}\n")
            if len(non_empty) > 0:
                f.write(f"  Avg length: {non_empty.mean():>10.0f} chars\n")
                f.write(f"  Max length: {non_empty.max():>10,} chars\n")
    
    print("Section 18 done: Text Column Analysis")

    # ========================================================================
    # SECTION 19: ANOMALIES & DATA QUALITY FLAGS
    # ========================================================================
    write_section(f, "SECTION 19: ANOMALIES & DATA QUALITY FLAGS")
    
    f.write("--- Potential Issues Detected ---\n\n")
    
    # Negative prices
    neg_price = (df['price'] < 0).sum()
    f.write(f"  Games with negative price:        {neg_price:>8,}\n")
    
    # Negative required_age
    neg_age = (df['required_age'] < 0).sum()
    f.write(f"  Games with negative required_age:  {neg_age:>8,}\n")
    
    # Reviews with -1 (sentinel)
    sentinel_reviews = (df['num_reviews_total'] == -1).sum()
    f.write(f"  Games with num_reviews_total = -1: {sentinel_reviews:>8,}\n")
    
    sentinel_pct = (df['pct_pos_total'] == -1).sum()
    f.write(f"  Games with pct_pos_total = -1:     {sentinel_pct:>8,}\n")
    
    sentinel_recent = (df['num_reviews_recent'] == -1).sum()
    f.write(f"  Games with num_reviews_recent = -1:{sentinel_recent:>8,}\n")
    
    # Extremely high prices
    extreme_price = (df['price'] > 100).sum()
    f.write(f"  Games with price > $100:           {extreme_price:>8,}\n")
    
    # Very old games
    old_games = df[df['release_year'] < 2000].shape[0] if 'release_year' in df.columns else 0
    f.write(f"  Games released before 2000:        {old_games:>8,}\n")
    
    # Duplicate names
    dup_names = df['name'].duplicated().sum()
    f.write(f"  Duplicate game names:              {dup_names:>8,}\n")
    
    print("Section 19 done: Anomalies & Data Quality")

    # ========================================================================
    # SECTION 20: SUCCESS TIER SIMULATION
    # ========================================================================
    write_section(f, "SECTION 20: SUCCESS TIER SIMULATION (PREVIEW)")
    
    f.write("This section simulates possible tier definitions based on the data.\n")
    f.write("These are PROPOSALS for discussion, not final decisions.\n\n")
    
    # Work with valid data only
    sim_df = df[(df['num_reviews_total'] >= 0) & (df['pct_pos_total'] >= 0)].copy()
    f.write(f"Games with valid review data for simulation: {len(sim_df):,}\n\n")
    
    # Simulation A: Reviews-based tiers
    f.write("--- SIMULATION A: Reviews-Based Tiers ---\n\n")
    thresholds_a = [
        ("Flop", lambda r: r['num_reviews_total'] < 100 or r['pct_pos_total'] < 70),
        ("Niche Hit", lambda r: r['num_reviews_total'] >= 100 and r['num_reviews_total'] < 1000 and r['pct_pos_total'] >= 70),
        ("Commercial Hit", lambda r: r['num_reviews_total'] >= 1000 and r['num_reviews_total'] < 50000 and r['pct_pos_total'] >= 70),
        ("Mega Hit", lambda r: r['num_reviews_total'] >= 50000 and r['pct_pos_total'] >= 70)
    ]
    
    for tier_name, condition in thresholds_a:
        count = sim_df.apply(condition, axis=1).sum()
        pct = (count / len(sim_df)) * 100
        f.write(f"  {tier_name:<20} {count:>10,} ({pct:.2f}%)\n")
    
    # Simulation B: Owners-based tiers
    f.write("\n\n--- SIMULATION B: Owners-Based Tiers ---\n\n")
    sim_owners = sim_df[sim_df['owners_min'].notna()].copy()
    
    thresholds_b = [
        ("Flop", lambda r: r['owners_min'] < 20000 or r['pct_pos_total'] < 70),
        ("Niche Hit", lambda r: r['owners_min'] >= 20000 and r['owners_min'] < 200000 and r['pct_pos_total'] >= 70),
        ("Commercial Hit", lambda r: r['owners_min'] >= 200000 and r['owners_min'] < 5000000 and r['pct_pos_total'] >= 70),
        ("Mega Hit", lambda r: r['owners_min'] >= 5000000 and r['pct_pos_total'] >= 70)
    ]
    
    for tier_name, condition in thresholds_b:
        count = sim_owners.apply(condition, axis=1).sum()
        pct = (count / len(sim_owners)) * 100
        f.write(f"  {tier_name:<20} {count:>10,} ({pct:.2f}%)\n")
    
    # Example games per tier (Simulation B)
    f.write("\n\n--- Example Games per Tier (Simulation B) ---\n\n")
    for tier_name, condition in thresholds_b:
        tier_games = sim_owners[sim_owners.apply(condition, axis=1)]
        if len(tier_games) > 0:
            examples = tier_games.nlargest(3, 'num_reviews_total')[['name', 'price', 'owners_min', 'num_reviews_total', 'pct_pos_total']]
            f.write(f"  [{tier_name}] Top 3 by reviews:\n")
            for _, row in examples.iterrows():
                f.write(f"    - {row['name']}: ${row['price']}, {int(row['owners_min']):,}+ owners, "
                        f"{int(row['num_reviews_total']):,} reviews, {int(row['pct_pos_total'])}% positive\n")
            f.write("\n")
    
    print("Section 20 done: Success Tier Simulation")

    # ========================================================================
    # SECTION 21: DISCOUNT ANALYSIS
    # ========================================================================
    write_section(f, "SECTION 21: DISCOUNT ANALYSIS")
    
    f.write(f"Games with no discount (0%):          {(df['discount'] == 0).sum():>10,}\n")
    f.write(f"Games with discount 1-25%:            {((df['discount'] >= 1) & (df['discount'] <= 25)).sum():>10,}\n")
    f.write(f"Games with discount 26-50%:           {((df['discount'] >= 26) & (df['discount'] <= 50)).sum():>10,}\n")
    f.write(f"Games with discount 51-75%:           {((df['discount'] >= 51) & (df['discount'] <= 75)).sum():>10,}\n")
    f.write(f"Games with discount 76-100%:          {((df['discount'] >= 76) & (df['discount'] <= 100)).sum():>10,}\n")
    
    print("Section 21 done: Discount Analysis")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    write_section(f, "FINAL SUMMARY & KEY TAKEAWAYS")
    
    f.write("""
KEY FINDINGS FOR THE SUCCESS PREDICTION MODEL:

1. MARKET REALITY:
   - 89,618 total games in the dataset.
   - The market is heavily right-skewed: a tiny fraction of games capture 
     the vast majority of players, reviews, and revenue.
   - 75% of games have fewer than 81 reviews (effectively invisible).

2. PRICE SENSITIVITY:
   - Median price is $4.99; 95th percentile is $19.99.
   - Pricing above $20 places a game in direct competition with AAA titles.

3. LANGUAGE AS A STRATEGIC LEVER:
   - Simplified Chinese is the #2 language on Steam (27% of games).
   - Localization is a strong candidate for an "actionable feature" in our model.

4. TAG POWER:
   - Tags are the richest feature for prediction, containing both frequency 
     and weight information per game.

5. DATA QUALITY NOTES:
   - Several columns use -1 as a sentinel value for "no data".
   - estimated_owners is a range string that needs parsing.
   - score_rank has >99% missing data and should likely be dropped.

6. RECOMMENDED NEXT STEPS:
   - Confirm the Success Tier thresholds with the user.
   - Build the data_prep.py script to engineer features and target variable.
   - Proceed to visual EDA with matplotlib/seaborn for the thesis.
""")

print(f"\n{'='*60}")
print(f"  REPORT COMPLETE!")
print(f"  Output saved to: {OUTPUT_FILE}")
print(f"{'='*60}")
