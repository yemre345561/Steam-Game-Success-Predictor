import pandas as pd

df = pd.read_csv('model_ready_dataset.csv')
target_col = 'target_tier'
review_metrics = ['pct_pos_total', 'num_reviews_total', 'peak_ccu']
temporal_leak = ['release_year']
all_features = [c for c in df.columns if c not in [target_col, 'target_3class']]
structural_features = [c for c in all_features if c not in review_metrics + temporal_leak]

print(f"Total structural features: {len(structural_features)}")
print()

# What UI provides
ui_numerical = ['price', 'achievements', 'dlc_count', 'tag_count', 'screenshot_count', 'required_age']
ui_binary = ['is_early_access', 'has_controller', 'has_trading_cards', 'has_workshop', 'has_website', 'supports_mac', 'supports_linux']
ui_languages = ['lang_chinese', 'lang_japanese', 'lang_german', 'lang_french', 'lang_russian', 'lang_korean', 'lang_brazilian_pt', 'lang_spanish']
ui_genres = ['genre_indie', 'genre_action', 'genre_adventure', 'genre_rpg', 'genre_strategy', 'genre_simulation', 'genre_casual', 'genre_sports', 'genre_racing', 'genre_free_to_play']
ui_tags = ['tag_multiplayer', 'tag_singleplayer', 'tag_action', 'tag_co_op', 'tag_adventure', 'tag_great_soundtrack', 'tag_open_world', 'tag_atmospheric', 'tag_rpg', 'tag_story_rich', 'tag_strategy', 'tag_free_to_play', 'tag_online_co_op', 'tag_first_person', 'tag_shooter', 'tag_sandbox', 'tag_third_person', 'tag_simulation', 'tag_funny', 'tag_difficult', 'tag_casual', 'tag_2d_platformer', 'tag_linear', 'tag_hidden_object', 'tag_minimalist']

# Auto-derived
auto_derived = ['is_free', 'has_achievements', 'has_dlc', 'has_multiplayer', 'has_coop', 'lang_count']

all_ui = ui_numerical + ui_binary + ui_languages + ui_genres + ui_tags + auto_derived

print(f"UI covers: {len(all_ui)} features")
print()

missing = [f for f in structural_features if f not in all_ui]
extra = [f for f in all_ui if f not in structural_features]

if missing:
    print(f"MISSING from UI ({len(missing)}):")
    for f in missing:
        print(f"  - {f}")
else:
    print("NO MISSING FEATURES")

if extra:
    print(f"\nEXTRA in UI ({len(extra)}):")
    for f in extra:
        print(f"  - {f}")
else:
    print("NO EXTRA FEATURES")

print(f"\n--- All 62 features mapped ---")
print(f"  Numerical (user input):  {len(ui_numerical)} -> {ui_numerical}")
print(f"  Binary (user toggle):    {len(ui_binary)} -> {ui_binary}")
print(f"  Languages (user toggle): {len(ui_languages)}")
print(f"  Genres (user toggle):    {len(ui_genres)}")
print(f"  Tags (user toggle):      {len(ui_tags)}")
print(f"  Auto-derived:            {len(auto_derived)} -> {auto_derived}")
print(f"  TOTAL:                   {len(all_ui)}")
