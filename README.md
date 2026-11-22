# Steam Game Success Predictor

Bachelor's thesis, Yeditepe University, Department of Information Systems and Technologies, 2026.

**Predicting Game Success and Identifying Critical Performance Drivers on the Steam Platform: A Machine Learning Approach**

The project asks whether a Steam game's commercial outcome can be estimated from store and design features a developer can set, without using review counts or player counts as inputs. The primary model is a Random Forest classifier trained on 89,618 games and 62 structural features. The label is the lower bound of Steam's estimated owner range. The rows are released games in the cleaned March 2025 table, so fields such as DLC count, achievements, and screenshots are the values stored in that snapshot.

Supervisor: Prof. Dr. Aşkın Demirağ.

## Key results

These are the recorded test results for the primary setup: a 3-class target, 62 structural features, a stratified 80/20 split, and `random_state=42`. The same run is shown in `figures/experiment_comparison.png`, `figures/cv_stability.png`, and `figures/confusion_matrix.png`. Re-running the scripts requires `model_ready_dataset.csv`, which is not in this repository.

| Metric | Value |
|--------|-------|
| Accuracy | 77.44% |
| Macro-F1 | 63.48% |
| Cross-validation Macro-F1 | 63.52% ± 0.32% |
| Features in the primary model | 62 structural |
| Dataset | 89,618 Steam games |

On the 3-class label the cleaned table has 67,738 Flop games (75.59%, under 20,000 owners), 17,240 Niche Hit games (19.24%, 20,000 to under 200,000), and 4,640 Successful games (5.18%, 200,000 or more). Accuracy sits above Macro-F1 because Flop is most of the table. The forest uses `class_weight='balanced'`.

The project cites the [Steam Games Dataset (Artermiloff 2025) on Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) as the public source. The preprocessing script expects `games_march2025_cleaned.csv`, the cleaned table used for the recorded results (89,618 rows, 47 columns).

## Feature sets compared

`model_training.py` fits the same Random Forest four times.

The 4-class label is Flop (under 20,000 owners), Niche Hit (20,000 to under 200,000), Commercial Hit (200,000 to under 5,000,000), and Mega Hit (5,000,000 or more). The 3-class label merges Commercial Hit and Mega Hit into Successful.

The 66-feature runs use every column except the label. That set adds positive-review percentage, review count, peak concurrent users, and release year on top of the 62 structural features.

| Experiment | Setup | Accuracy | Macro-F1 |
|:---:|---|:---:|:---:|
| 1 | 4-class, 66 features | 85.99% | 75.16% |
| 2 | 4-class, 62 structural features | 79.73% | 52.28% |
| 3 | 3-class, 66 features | 86.06% | 78.04% |
| 4 (primary) | 3-class, 62 structural features | 77.44% | 63.48% |

Experiment 4 is the primary model. Its Macro-F1 is 14.56 points below Experiment 3 (63.48% compared with 78.04%). Those four extra columns account for the gap. Review percentage, review count, and peak concurrent users are observed after release, and release year is temporal metadata, so the primary model keeps them out of its feature set. The comparison chart rounds this same run to one decimal place.

## Top success drivers

Recorded importances for the primary model, in the same order as `figures/feature_importance.png`:

| Rank | Feature | Importance |
|:---:|---------|:---:|
| 1 | Tag count | 9.50% |
| 2 | Trading cards | 8.80% |
| 3 | Price | 6.40% |
| 4 | Achievements | 6.00% |
| 5 | Screenshot count | 4.80% |
| 6 | Language count | 4.70% |

The `figures/` folder also contains `cv_stability.png`, `target_distribution.png`, `market_saturation.png`, `binary_features_heatmap.png`, `category_contribution.png`, and `confusion_matrix.png`.

## Repository layout

```
data_preprocessing.py    # games_march2025_cleaned.csv -> model_ready_dataset.csv
model_training.py        # four Random Forest experiments, metrics, and importances
model_optimization.py    # 5-fold cross-validation, grid search, decision tree comparison
test_predictions.py      # posts four presets to a running demo
predictor_app.py         # Flask demo; trains the primary model at startup
templates/predictor.html
static/predictor_style.css
figures/
```

The grid search in `model_optimization.py` is a separate comparison. It tries 200 or 400 trees and `max_depth` of `None`, 30, or 50. The primary forest stays on the fixed settings in `model_training.py` and `predictor_app.py`: 300 trees, `max_depth=25`, `min_samples_split=10`, `min_samples_leaf=4`, `max_features='sqrt'`, `class_weight='balanced'`, and `random_state=42`.

## Running the project

`games_march2025_cleaned.csv` and `model_ready_dataset.csv` are not stored in this repository. Both names are listed in `.gitignore` with the other large data files. `.gitignore` also lists `steam_games_data.csv`. The scripts never open that name.

The published code can rebuild `model_ready_dataset.csv` from the cleaned input. It has no script that rebuilds `games_march2025_cleaned.csv` from the Kaggle download. That earlier cleaning step is not published here.

1. Place `games_march2025_cleaned.csv` in the working directory. `data_preprocessing.py` reads that filename. The recorded file has 89,618 rows and 47 columns.
2. Install the packages the scripts import, then build the model-ready file:

```bash
pip install pandas numpy scikit-learn flask
python data_preprocessing.py
```

3. Train and print the four experiments:

```bash
python model_training.py
```

4. Run cross-validation and the grid search:

```bash
python model_optimization.py
```

5. Start the Flask demo. On startup it loads `model_ready_dataset.csv`, fits the Experiment 4 forest, and runs 5-fold stratified cross-validation before it serves the page:

```bash
python predictor_app.py
# http://127.0.0.1:5000
```

The demo includes six presets: AAA, small indie, free-to-play, mid-tier RPG, simulation builder, and horror indie. The form edits the visible inputs. The page derives the free-to-play, achievement, DLC, multiplayer, and co-op flags from those inputs, and it sets language count to the number of selected language flags plus one. When a request omits them, screenshot count defaults to 7 and required age defaults to 0. The reply is Flop, Niche Hit, or Successful, with class probabilities, short strengths and weaknesses, and a comparison against class averages. Change the inputs and predict again to try another profile.

`test_predictions.py` is a client for that server. It needs the `requests` package and a demo that is already running at `http://localhost:5000`.

`model_training.py`, `model_optimization.py`, and `predictor_app.py` read `model_ready_dataset.csv` as soon as they start. They stop if the file is missing.

## The 62 structural features

| Group | Count | Columns |
|----------|:-----:|---------|
| Numerical | 7 | price, achievements, dlc_count, required_age, lang_count, tag_count, screenshot_count |
| Binary | 12 | is_free, has_achievements, has_dlc, is_early_access, has_multiplayer, has_coop, has_controller, has_trading_cards, has_workshop, has_website, supports_mac, supports_linux |
| Languages | 8 | Chinese, Japanese, German, French, Russian, Korean, Brazilian Portuguese, Spanish |
| Genres | 10 | Indie, Action, Adventure, RPG, Strategy, Simulation, Casual, Sports, Racing, Free to Play |
| Tags | 25 | 25 Steam tags selected in `data_preprocessing.py` |

`model_ready_dataset.csv` has 67 columns. Besides the 62 structural features and `target_tier`, it stores `release_year`, `peak_ccu`, `pct_pos_total`, and `num_reviews_total` for the 66-feature experiments.

## Methodology

1. Start from `games_march2025_cleaned.csv`: 89,618 rows and 47 columns.
2. Drop the 25 columns named in `data_preprocessing.py`: sparse text and score fields, media URLs, descriptions, developer and publisher names, row identifiers, redundant score fields, discount, and playtime. The list includes columns with a missing rate above 90% (`score_rank`, `metacritic_url`) and columns below that rate (`reviews` at 88.39%, `notes` at 81.43%). `metacritic_score` is dropped as a mostly zero score field.
3. Replace the sentinel value -1 with 0 in `pct_pos_total` and `num_reviews_total` (36,419 rows). The pipeline keeps no separate "has reviews" flag.
4. Build `target_tier` from the lower bound of `estimated_owners`. Derive `release_year` from `release_date` and keep it in the file for the 66-feature runs. The training code then excludes release year from the 62-feature model and treats it as temporal metadata.
5. Add the structural features above and write `model_ready_dataset.csv` (89,618 rows, 67 columns).
6. Fit the Random Forest settings listed above. `model_training.py` scores a stratified 80/20 split, prints a confusion matrix and the top importances, and repeats that for all four experiments. `model_optimization.py` runs 5-fold stratified cross-validation with `shuffle=True` and `random_state=42`.

## Tech stack

The published scripts use the Python standard library, pandas, NumPy, scikit-learn, and Flask. `test_predictions.py` also imports `requests`. The repository has no pinned dependency file. The charts in `figures/` are stored PNG files. matplotlib and seaborn are absent from the scripts in this tree.

## Citation

Açıkoğlu, Y. (2026). Predicting Game Success and Identifying Critical Performance Drivers on the Steam Platform: A Machine Learning Approach. Bachelor's Thesis, Yeditepe University, Department of Information Systems and Technologies.

## Author

Yunusemre Açıkoğlu  
[GitHub](https://github.com/yemre345561) · [LinkedIn](https://www.linkedin.com/in/yunus-emre-a%C3%A7%C4%B1ko%C4%9Flu-1479b1203)
