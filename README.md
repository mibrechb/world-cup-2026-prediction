# World Cup 2026 Match & Tipp Prediction

A lightweight adaptation of Javier Ruano's [World Cup 2026 prediction model](https://github.com/javierruanohdez/world-cup-2026-prediction).

This project reuses the original model and data pipeline, and adds custom functions for per-match score simulations and optimal tip recommendations for the [SRF Sport FIFA World Cup 2026 Tippspiel](https://wmtippspiel.srf.ch/).

Predictions are based on a supervised [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html), which treats each match as a win-draw-loss classification problem and returns the associated probabilities. The model is trained on team-level and match-level features from international match results, FIFA rankings, tournament performance, and recent form.

This adaptation uses those probability outputs to simulate likely scorelines. Goals are sampled with a Poisson-based score model using round-dependent assumptions for average total goals. Instead of simulating only complete tournaments, this project repeatedly simulates individual matchups and uses the results to estimate score probabilities and expected tip points.

In short, this adaptation focuses on:

- simulating individual match outcomes,
- estimating likely scorelines,
- ranking possible score tips by expected points based on SRF Sport tipping-game rules,
- generating a full matchup prediction matrix.


## SRF Sport tipping game rules

Tipping recommendations are optimised based on expected points, i.e. the probability of each simulated outcome multiplied by the points awarded under the [SRF Sport FIFA World Cup 2026 tipping-game scoring rules](https://wmtippspiel.srf.ch/info/rules).

The tournament phases are named as follows:

- `group` uses group-stage scoring.
- `play-in`, `round_of_16`, `quarterfinal`, `semifinal`, and `final` use K.O.-phase scoring.

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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mibrechb/world-cup-2026-prediction/main?urlpath=%2Fdoc%2Ftree%2F01_match_prediction.ipynb)

```text
/
├── 01_match_prediction.ipynb
├── 02_prediction_matrix.ipynb
├── utils/
│   └── simulation.py
├── data/
│   └── raw/
│   └── processed/
├── models/
│   └── gradient_boosting_v1.pkl
└── simulation_results/
    └── group/*.csv