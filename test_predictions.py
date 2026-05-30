import requests

# Test all presets with auto-derived features
tests = {
    'AAA Game': {
        'price': 59.99, 'achievements': 50, 'dlc_count': 3,
        'tag_count': 25, 'screenshot_count': 15, 'required_age': 18,
        'is_early_access': 0, 'has_controller': 1, 'has_trading_cards': 1,
        'has_workshop': 0, 'has_website': 1,
        'supports_mac': 0, 'supports_linux': 0,
        'lang_chinese': 1, 'lang_japanese': 1, 'lang_german': 1, 'lang_french': 1,
        'lang_russian': 1, 'lang_korean': 1, 'lang_brazilian_pt': 1, 'lang_spanish': 1,
        'genre_action': 1, 'genre_adventure': 1,
        'tag_multiplayer': 1, 'tag_singleplayer': 1, 'tag_action': 1, 'tag_co_op': 1,
        'tag_adventure': 1, 'tag_great_soundtrack': 1, 'tag_open_world': 1, 'tag_atmospheric': 1,
        'tag_story_rich': 1, 'tag_online_co_op': 1, 'tag_third_person': 1,
    },
    'Small Indie': {
        'price': 4.99, 'achievements': 0, 'dlc_count': 0,
        'tag_count': 8, 'screenshot_count': 4, 'required_age': 0,
        'has_workshop': 0, 'has_website': 0,
        'genre_indie': 1, 'genre_casual': 1,
        'tag_singleplayer': 1, 'tag_casual': 1, 'tag_2d_platformer': 1, 'tag_minimalist': 1,
    },
    'F2P': {
        'price': 0, 'achievements': 20, 'dlc_count': 5,
        'tag_count': 20, 'screenshot_count': 10, 'required_age': 0,
        'has_controller': 1, 'has_trading_cards': 1,
        'has_workshop': 0, 'has_website': 1,
        'lang_chinese': 1, 'lang_japanese': 1, 'lang_german': 1, 'lang_french': 1,
        'lang_russian': 1, 'lang_korean': 1, 'lang_brazilian_pt': 1, 'lang_spanish': 1,
        'genre_action': 1, 'genre_free_to_play': 1,
        'tag_multiplayer': 1, 'tag_action': 1, 'tag_co_op': 1, 'tag_free_to_play': 1,
        'tag_online_co_op': 1, 'tag_first_person': 1, 'tag_shooter': 1,
    },
    'Mid RPG': {
        'price': 19.99, 'achievements': 30, 'dlc_count': 1,
        'tag_count': 18, 'screenshot_count': 10, 'required_age': 16,
        'has_controller': 1, 'has_trading_cards': 1,
        'has_workshop': 0, 'has_website': 1,
        'lang_chinese': 1, 'lang_japanese': 1, 'lang_german': 1, 'lang_french': 1,
        'genre_indie': 1, 'genre_adventure': 1, 'genre_rpg': 1,
        'tag_singleplayer': 1, 'tag_adventure': 1, 'tag_great_soundtrack': 1,
        'tag_open_world': 1, 'tag_atmospheric': 1, 'tag_rpg': 1, 'tag_story_rich': 1,
        'tag_third_person': 1,
    },
}

# Simulate JS auto-derive
def auto_derive(data):
    data.setdefault('is_free', 1 if data.get('price', 0) == 0 else 0)
    data.setdefault('has_achievements', 1 if data.get('achievements', 0) > 0 else 0)
    data.setdefault('has_dlc', 1 if data.get('dlc_count', 0) > 0 else 0)
    data.setdefault('has_multiplayer', data.get('tag_multiplayer', 0))
    data.setdefault('has_coop', data.get('tag_co_op', 0))
    langs = ['lang_chinese','lang_japanese','lang_german','lang_french','lang_russian','lang_korean','lang_brazilian_pt','lang_spanish']
    data.setdefault('lang_count', sum(data.get(l, 0) for l in langs) + 1)
    return data

print("=" * 70)
print("PRESET TEST RESULTS (with auto-derived features)")
print("=" * 70)
for name, data in tests.items():
    payload = auto_derive(data.copy())
    r = requests.post('http://localhost:5000/predict', json=payload)
    res = r.json()
    if 'error' in res:
        print(f"{name:15s} => ERROR: {res['error']}")
        continue
    probs = res['probabilities']
    label = res['label']
    print(f"{name:15s} => {label:12s}  Flop:{probs['Flop']:5.1f}%  Niche:{probs['Niche Hit']:5.1f}%  Successful:{probs['Successful']:5.1f}%")
    print()
