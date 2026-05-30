"""
Steam Game Success Predictor - Web Demo
========================================
This is a STANDALONE demo app. It does NOT modify any existing thesis files.
It loads model_ready_dataset.csv, trains the exact same Random Forest model
from the thesis (Exp 4: 3-Class, Structural Only, Tuned), and serves
an interactive web interface for predictions.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import json
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ============================================================
#  MODEL TRAINING (Exact replica of thesis Exp 4)
# ============================================================
print("\n" + "="*60)
print("  STEAM SUCCESS PREDICTOR - Loading & Training...")
print("="*60)

# Load the thesis dataset
df = pd.read_csv('model_ready_dataset.csv')
print(f"  Dataset loaded: {len(df):,} games, {len(df.columns)} columns")

# Create 3-class target (same as thesis)
df['target_3class'] = df['target_tier'].apply(lambda x: min(x, 2))

# Define features (same as thesis Exp 4: structural only, no release_year)
target_col = 'target_tier'
review_metrics = ['pct_pos_total', 'num_reviews_total', 'peak_ccu']
temporal_leak = ['release_year']
all_features = [c for c in df.columns if c not in [target_col, 'target_3class']]
structural_features = [c for c in all_features if c not in review_metrics + temporal_leak]

X = df[structural_features]
y = df['target_3class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Train with EXACT thesis parameters (Exp 4: 62 structural features)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='macro')

# Per-class metrics
class_labels = ['Flop', 'Niche Hit', 'Successful']
report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Cross-validation (StratifiedKFold — same as thesis figures)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)

# Feature importance
importances = pd.Series(model.feature_importances_, index=structural_features).sort_values(ascending=False)

# Compute success-class averages for comparison
success_avg = df[df['target_3class'] == 2][structural_features].mean()
flop_avg = df[df['target_3class'] == 0][structural_features].mean()

print(f"  Model trained! Accuracy: {test_acc:.2%} | Macro-F1: {test_f1:.2%}")
print(f"  Cross-Val F1: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
print("  Ready to serve predictions!\n")

# ============================================================
#  FEATURE METADATA (for UI and insights)
# ============================================================
# Hidden features: hardcoded in predict route
HIDDEN_DEFAULTS = {
    'screenshot_count': 7,   # Dataset median — auto-applied, not shown in UI
    'required_age': 0,       # Most common value — auto-applied, not shown in UI
}

FEATURE_GROUPS = {
    'numerical': {
        'label': 'Basic Info',
        'icon': '💰',
        'features': {
            'price': {'label': 'Price ($)', 'min': 0, 'max': 200, 'step': 0.01, 'default': 9.99},
            'achievements': {'label': 'Achievements', 'min': 0, 'max': 500, 'step': 1, 'default': 0},
            'dlc_count': {'label': 'Number of DLCs', 'min': 0, 'max': 20, 'step': 1, 'default': 0},
            'tag_count': {'label': 'Number of Steam Tags', 'min': 0, 'max': 40, 'step': 1, 'default': 10},
        }
    },
    'binary': {
        'label': 'Game Properties',
        'icon': '🎮',
        'features': {
            'is_early_access': {'label': 'Early Access'},
            'has_controller': {'label': 'Controller Support'},
            'has_trading_cards': {'label': 'Trading Cards'},
            'has_workshop': {'label': 'Steam Workshop'},
            'has_website': {'label': 'Official Website'},
            'supports_mac': {'label': 'Mac Support'},
            'supports_linux': {'label': 'Linux Support'},
        }
    },
    'languages': {
        'label': 'Language Support',
        'icon': '🌍',
        'features': {
            'lang_chinese': {'label': 'Chinese'},
            'lang_japanese': {'label': 'Japanese'},
            'lang_german': {'label': 'German'},
            'lang_french': {'label': 'French'},
            'lang_russian': {'label': 'Russian'},
            'lang_korean': {'label': 'Korean'},
            'lang_brazilian_pt': {'label': 'Brazilian Portuguese'},
            'lang_spanish': {'label': 'Spanish'},
        }
    },
    'genres': {
        'label': 'Game Genres',
        'icon': '🏷️',
        'features': {
            'genre_indie': {'label': 'Indie'},
            'genre_action': {'label': 'Action'},
            'genre_adventure': {'label': 'Adventure'},
            'genre_rpg': {'label': 'RPG'},
            'genre_strategy': {'label': 'Strategy'},
            'genre_simulation': {'label': 'Simulation'},
            'genre_casual': {'label': 'Casual'},
            'genre_sports': {'label': 'Sports'},
            'genre_racing': {'label': 'Racing'},
            'genre_free_to_play': {'label': 'Free to Play'},
        }
    },
    'tags': {
        'label': 'Steam Tags',
        'icon': '🔖',
        'features': {
            'tag_multiplayer': {'label': 'Multiplayer'},
            'tag_singleplayer': {'label': 'Singleplayer'},
            'tag_action': {'label': 'Action'},
            'tag_co_op': {'label': 'Co-Op'},
            'tag_adventure': {'label': 'Adventure'},
            'tag_great_soundtrack': {'label': 'Great Soundtrack'},
            'tag_open_world': {'label': 'Open World'},
            'tag_atmospheric': {'label': 'Atmospheric'},
            'tag_rpg': {'label': 'RPG'},
            'tag_story_rich': {'label': 'Story Rich'},
            'tag_strategy': {'label': 'Strategy'},
            'tag_free_to_play': {'label': 'Free to Play'},
            'tag_online_co_op': {'label': 'Online Co-Op'},
            'tag_first_person': {'label': 'First Person'},
            'tag_shooter': {'label': 'Shooter'},
            'tag_sandbox': {'label': 'Sandbox'},
            'tag_third_person': {'label': 'Third Person'},
            'tag_simulation': {'label': 'Simulation'},
            'tag_funny': {'label': 'Funny'},
            'tag_difficult': {'label': 'Difficult'},
            'tag_casual': {'label': 'Casual'},
            'tag_2d_platformer': {'label': '2D Platformer'},
            'tag_linear': {'label': 'Linear'},
            'tag_hidden_object': {'label': 'Hidden Object'},
            'tag_minimalist': {'label': 'Minimalist'},
        }
    }
}

# Presets
PRESETS = {
    'aaa_game': {
        'name': '🎮 Typical AAA Game',
        'desc': 'Big-budget, multi-platform, full localization',
        'values': {
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
        }
    },
    'small_indie': {
        'name': '🕹️ Small Indie Game',
        'desc': 'Low-budget solo dev, minimal features',
        'values': {
            'price': 4.99, 'achievements': 0, 'dlc_count': 0,
            'tag_count': 8, 'screenshot_count': 4, 'required_age': 0,
            'is_early_access': 0, 'has_controller': 0, 'has_trading_cards': 0,
            'has_workshop': 0, 'has_website': 0,
            'supports_mac': 0, 'supports_linux': 0,
            'genre_indie': 1, 'genre_casual': 1,
            'tag_singleplayer': 1, 'tag_casual': 1, 'tag_2d_platformer': 1, 'tag_minimalist': 1,
        }
    },
    'f2p_game': {
        'name': '💰 Free-to-Play Game',
        'desc': 'Free game with multiplayer focus',
        'values': {
            'price': 0, 'achievements': 20, 'dlc_count': 5,
            'tag_count': 20, 'screenshot_count': 10, 'required_age': 0,
            'is_early_access': 0, 'has_controller': 1, 'has_trading_cards': 1,
            'has_workshop': 0, 'has_website': 1,
            'supports_mac': 0, 'supports_linux': 0,
            'lang_chinese': 1, 'lang_japanese': 1, 'lang_german': 1, 'lang_french': 1,
            'lang_russian': 1, 'lang_korean': 1, 'lang_brazilian_pt': 1, 'lang_spanish': 1,
            'genre_action': 1, 'genre_free_to_play': 1,
            'tag_multiplayer': 1, 'tag_action': 1, 'tag_co_op': 1, 'tag_free_to_play': 1,
            'tag_online_co_op': 1, 'tag_first_person': 1, 'tag_shooter': 1,
        }
    },
    'rpg_mid': {
        'name': '⚔️ Mid-Tier RPG',
        'desc': 'Story-driven RPG with moderate budget',
        'values': {
            'price': 19.99, 'achievements': 30, 'dlc_count': 1,
            'tag_count': 18, 'screenshot_count': 10, 'required_age': 16,
            'is_early_access': 0, 'has_controller': 1, 'has_trading_cards': 1,
            'has_workshop': 0, 'has_website': 1,
            'supports_mac': 0, 'supports_linux': 0,
            'lang_chinese': 1, 'lang_japanese': 1, 'lang_german': 1, 'lang_french': 1,
            'genre_indie': 1, 'genre_adventure': 1, 'genre_rpg': 1,
            'tag_singleplayer': 1, 'tag_adventure': 1, 'tag_great_soundtrack': 1,
            'tag_open_world': 1, 'tag_atmospheric': 1, 'tag_rpg': 1, 'tag_story_rich': 1,
            'tag_third_person': 1,
        }
    },
    'sim_builder': {
        'name': '🏗️ Simulation Builder',
        'desc': 'City builder / tycoon with mod support',
        'values': {
            'price': 14.99, 'achievements': 15, 'dlc_count': 0,
            'tag_count': 15, 'screenshot_count': 8, 'required_age': 0,
            'is_early_access': 1, 'has_controller': 0, 'has_trading_cards': 0,
            'has_workshop': 1, 'has_website': 0,
            'supports_mac': 0, 'supports_linux': 1,
            'lang_german': 1, 'lang_french': 1,
            'genre_indie': 1, 'genre_simulation': 1, 'genre_strategy': 1,
            'tag_singleplayer': 1, 'tag_simulation': 1, 'tag_strategy': 1, 'tag_sandbox': 1,
        }
    },
    'horror_indie': {
        'name': '👻 Horror Indie',
        'desc': 'Atmospheric horror with story focus',
        'values': {
            'price': 9.99, 'achievements': 10, 'dlc_count': 0,
            'has_controller': 1, 'has_trading_cards': 0,
            'lang_chinese': 1, 'lang_russian': 1,
            'genre_indie': 1, 'genre_adventure': 1,
            'tag_singleplayer': 1, 'tag_atmospheric': 1, 'tag_story_rich': 1,
            'tag_first_person': 1, 'tag_difficult': 1,
        }
    }
}

# ============================================================
#  INSIGHT GENERATOR
# ============================================================
def generate_insights(features_dict, proba):
    """Generate smart insights based on the prediction and feature importance."""
    pred_class = int(np.argmax(proba))
    insights = {'strengths': [], 'weaknesses': [], 'recommendations': []}
    
    # Key binary features and their impact (from feature importance + success averages)
    key_checks = [
        ('has_dlc', 'DLC Content', success_avg['has_dlc']*100, 'Adding DLC content shows long-term commitment and increases perceived value.'),
        ('has_multiplayer', 'Multiplayer Support', success_avg['has_multiplayer']*100, 'Consider adding multiplayer — it dramatically increases player engagement and visibility.'),
        ('has_achievements', 'Achievement System', success_avg['has_achievements']*100, 'Adding achievements increases playtime and player retention.'),
        ('has_trading_cards', 'Steam Trading Cards', success_avg['has_trading_cards']*100, 'Trading cards attract collectors and boost store visibility.'),
        ('has_controller', 'Controller Support', success_avg['has_controller']*100, 'Controller support opens the game to couch and Steam Deck players.'),
        ('has_coop', 'Co-Op Mode', success_avg['has_coop']*100, 'Co-op functionality drives word-of-mouth and organic growth.'),
    ]
    
    for feat, label, success_pct, rec in key_checks:
        if features_dict.get(feat, 0) == 1:
            insights['strengths'].append(f"{label} is enabled — present in {success_pct:.0f}% of successful games.")
        elif success_pct > 40:
            insights['weaknesses'].append(f"No {label} — but {success_pct:.0f}% of successful games have it.")
            insights['recommendations'].append(rec)
    
    # Language check
    lang_feats = [f for f in features_dict if f.startswith('lang_')]
    lang_count = sum(features_dict.get(f, 0) for f in lang_feats)
    avg_success_langs = success_avg[['lang_chinese','lang_japanese','lang_german','lang_french','lang_russian','lang_korean','lang_brazilian_pt','lang_spanish']].sum()
    if lang_count < 3:
        insights['weaknesses'].append(f"Only {lang_count} major language(s) supported — successful games average {avg_success_langs:.0f} major languages.")
        insights['recommendations'].append("Prioritize adding German, French, and Chinese — these are the top 3 languages by market impact.")
    else:
        insights['strengths'].append(f"{lang_count} major languages supported — strong localization.")
    
    # Price insight
    price = features_dict.get('price', 0)
    avg_success_price = success_avg['price']
    if price < 2 and features_dict.get('is_free', 0) == 0:
        insights['weaknesses'].append(f"Price ${price:.2f} is very low — may signal low quality to buyers.")
        insights['recommendations'].append(f"Successful games average ${avg_success_price:.2f}. Consider pricing between $9.99-$19.99.")
    elif price > 0:
        insights['strengths'].append(f"Price point ${price:.2f} — successful games average ${avg_success_price:.2f}.")
    
    # Comparison with success averages
    comparison = {}
    compare_feats = ['price', 'achievements', 'dlc_count', 'lang_count']
    for f in compare_feats:
        comparison[f] = {
            'your_value': features_dict.get(f, 0),
            'success_avg': round(success_avg[f], 1),
            'flop_avg': round(flop_avg[f], 1)
        }
    
    return insights, comparison


# ============================================================
#  MODEL PERFORMANCE DATA (for reliability panel)
# ============================================================
MODEL_PERF = {
    'accuracy': round(test_acc * 100, 2),
    'macro_f1': round(test_f1 * 100, 2),
    'cv_mean': round(cv_scores.mean() * 100, 2),
    'cv_std': round(cv_scores.std() * 100, 2),
    'per_class': {},
    'confusion_matrix': cm.tolist(),
    'class_labels': class_labels,
    'top_features': [
        {'name': name, 'importance': round(val * 100, 2)} 
        for name, val in importances[:10].items()
    ],
    'dataset_size': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test),
}

for label in class_labels:
    MODEL_PERF['per_class'][label] = {
        'precision': round(report[label]['precision'] * 100, 1),
        'recall': round(report[label]['recall'] * 100, 1),
        'f1': round(report[label]['f1-score'] * 100, 1),
        'support': int(report[label]['support']),
    }

# Build feature stats for UI hints
FEATURE_STATS = {}
for feat in structural_features:
    imp = importances.get(feat, 0)
    FEATURE_STATS[feat] = {
        'importance': round(imp * 100, 2),
        'success_avg': round(success_avg[feat], 2),
        'flop_avg': round(flop_avg[feat], 2),
    }

# Top 15 most important features for UI badges
TOP_FEATURES = importances[:15].index.tolist()


# ============================================================
#  FLASK ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('predictor.html', 
                         feature_groups=FEATURE_GROUPS,
                         presets=PRESETS,
                         model_perf=MODEL_PERF,
                         feature_stats=FEATURE_STATS,
                         top_features=TOP_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Inject hidden defaults for features not shown in UI
        for k, v in HIDDEN_DEFAULTS.items():
            if k not in data:
                data[k] = v
        
        # Build feature vector in correct column order
        feature_vector = []
        for feat in structural_features:
            feature_vector.append(float(data.get(feat, 0)))
        
        X_input = pd.DataFrame([feature_vector], columns=structural_features)
        
        # Predict
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        
        # Generate insights
        insights, comparison = generate_insights(data, probabilities)
        
        result = {
            'prediction': int(prediction),
            'label': class_labels[prediction],
            'probabilities': {
                'Flop': round(float(probabilities[0]) * 100, 1),
                'Niche Hit': round(float(probabilities[1]) * 100, 1),
                'Successful': round(float(probabilities[2]) * 100, 1),
            },
            'insights': insights,
            'comparison': comparison,
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info')
def model_info():
    return jsonify(MODEL_PERF)

@app.route('/presets')
def get_presets():
    return jsonify(PRESETS)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Steam Success Predictor is LIVE!")
    print("  Open your browser: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, port=5000)
