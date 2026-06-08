# World Cup 2026 Match & Tipp Prediction

This repository contains the code used to generate an interactive World Cup 2026 prediction portal, hosted via GitHub Pages.

[![Interactive Portal](https://img.shields.io/badge/Click%20me-%E2%9A%BD%20Interactive%20Prediction%20Portal-1f6feb?style=for-the-badge)](https://mibrechb.github.io/world-cup-2026-prediction/)

## How does it work

This project reuses the original outcome (win-draw-loss) model and data pipeline of Javier Ruano's [World Cup 2026 prediction model](https://github.com/javierruanohdez/world-cup-2026-prediction) as a first stage. It adds custom functions in a secondary stage for per-match score simulations and optimal tip recommendations for the [SRF Sport FIFA World Cup 2026 Tippspiel](https://wmtippspiel.srf.ch/).

*First stage outcome predictions* are based on a supervised [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), which treats each match as a win-draw-loss classification problem and returns the associated probabilities. The model is trained on team-level and match-level features from international match results, FIFA rankings, tournament performance, and recent form.

The *second stage goal prediction* uses per-class probability outputs from the outcome classifier to guide scoreline simulation. A second-stage goal layer was added and is based on two trained [GradientBoostingRegressors](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html): one for total goals and one for goal difference. The regressed predictions are converted into expected home and away goal rates, and final scoreline probabilities are generated from a Poisson-based sampler using those expected goals.

Instead of simulating only complete tournaments, this project repeatedly simulates individual matchups and uses the results to estimate score probabilities and expected tip points.

In short, this project focuses on:

- simulating individual match outcomes,
- estimating likely scorelines,
- ranking possible score tips by expected points based on SRF Sport tipping-game rules,
- generating a full matchup prediction matrix.


## SRF Sport tipping game rules

Tipping recommendations are optimised based on expected points, i.e. the probability of each simulated outcome multiplied by the points awarded under the [SRF Sport FIFA World Cup 2026 tipping-game scoring rules](https://wmtippspiel.srf.ch/info/rules).

The tournament phases are named as follows:

- `group` uses group-stage scoring.
- `round_of_32`, `round_of_16`, `quarterfinal`, `semifinal`, `third_place`, and `final` use K.O.-phase scoring.

For knockout matches, the result after 120 minutes is used for tipping-game scoring. Goals from a penalty shootout are not included.

| Rule | Group stage | K.O. phase | Bonus questions |
|---|---:|---:|---:|
| Correctly tipped winner or draw, independent of goals | 5 | 10 | — |
| Correct number of home-team goals | 1 | 2 | — |
| Correct number of away-team goals | 1 | 2 | — |
| Correct goal difference, if the winner/draw is correct | 3 | 6 | — |
| Correct World Cup winner prediction | — | — | 50 |
| Each additional correctly answered bonus question | — | — | 20 |


## How to use

The main workflow is split into two executable Jupyter notebooks. For easy usage, the notebooks can be launched directly in Binder.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mibrechb/world-cup-2026-prediction/main?urlpath=%2Fdoc%2Ftree%2F01_run-single-match-prediction.ipynb)

```text
/
├── 01_construct-goal-model.ipynb          # construct 2nd stage goal-prediction model
├── 02_run-single-match-prediction.ipynb   # single-match simulation and tipping analysis
├── 03_run-multi-match-prediction.ipynb    # full matchup matrix simulation
├── 04_generate-interactive-webpage.ipynb  # standalone interactive HTML generation
├── utils/
│   ├── simulation.py                      # shared simulation and score-sampling logic
│   └── lut_flag.json                      # team-to-flag lookup table
├── data/
│   ├── raw/                               # source datasets
│   └── processed/                         # cleaned and feature-engineered inputs
├── models/
│   └── gradient_boosting_v1.pkl           # trained outcome classifier
└── simulation_results/
    └── group/*.csv                        # exported matchup summaries and detail tables
```