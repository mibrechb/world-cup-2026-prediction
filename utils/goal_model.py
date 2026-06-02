from pathlib import Path
from math import lgamma

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from utils.outcome_model import (
    apply_modern_adjustment,
    prepare_feature_frame,
)
from utils.runtime_data import EMPIRICAL_MAX_GOALS, get_result_code, historical_results, matches, model

DEFAULT_VALIDATION_START_DATE = '2018-01-01'
GOAL_MODEL_RANDOM_STATE = 42
GOAL_PROFILE_WINDOW = 8
GOAL_TOTAL_MODEL_PATH = Path('models/model_grboostreg_goal-total_v1.pkl')
GOAL_DIFF_MODEL_PATH = Path('models/model_grboostreg_goal-diff_v1.pkl')

# Backward-compatible aliases for notebook imports.
GOAL_HOME_MODEL_PATH = GOAL_TOTAL_MODEL_PATH
GOAL_AWAY_MODEL_PATH = GOAL_DIFF_MODEL_PATH

GOAL_REGRESSION_FEATURE_COLUMNS = [
    'fifa_diff',
    'home_last5_win_rate',
    'away_last5_win_rate',
    'h2h_home_win_rate',
    'h2h_draw_rate',
    'h2h_matches_played',
    'home_penalty_win_rate',
    'away_penalty_win_rate',
    'neutral',
    'fifa_diff_x_home_form',
    'fifa_diff_x_away_form',
    'h2h_effective',
    'p_home',
    'p_draw',
    'p_away',
    'abs_fifa_diff',
    'home_recent_goals_for',
    'home_recent_goals_against',
    'home_recent_total_goals',
    'home_recent_btts_rate',
    'away_recent_goals_for',
    'away_recent_goals_against',
    'away_recent_total_goals',
    'away_recent_btts_rate',
    'goal_env_home',
    'goal_env_away',
    'goal_env_total',
]

GOAL_PROFILE_DEFAULTS = {
    'recent_goals_for': 1.35,
    'recent_goals_against': 1.35,
    'recent_total_goals': 2.70,
    'recent_btts_rate': 0.48,
}


def _build_team_goal_profiles(goal_history, window=GOAL_PROFILE_WINDOW):
    history = goal_history[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'total_goals']].copy()

    home_view = history.rename(
        columns={
            'home_team': 'team',
            'away_team': 'opponent',
            'home_score': 'goals_for',
            'away_score': 'goals_against',
        }
    )
    home_view['side'] = 'home'

    away_view = history.rename(
        columns={
            'away_team': 'team',
            'home_team': 'opponent',
            'away_score': 'goals_for',
            'home_score': 'goals_against',
        }
    )
    away_view['side'] = 'away'

    team_history = pd.concat([home_view, away_view], ignore_index=True)
    team_history = team_history.sort_values(['team', 'date', 'side']).reset_index(drop=True)
    team_history['both_teams_scored'] = (
        (team_history['goals_for'] > 0) & (team_history['goals_against'] > 0)
    ).astype(float)

    profile_columns = {
        'goals_for': 'recent_goals_for',
        'goals_against': 'recent_goals_against',
        'total_goals': 'recent_total_goals',
        'both_teams_scored': 'recent_btts_rate',
    }
    for source_column, profile_column in profile_columns.items():
        team_history[profile_column] = (
            team_history.groupby('team')[source_column]
            .transform(lambda series: series.shift(1).rolling(window=window, min_periods=2).mean())
        )
        team_history[profile_column] = team_history[profile_column].fillna(GOAL_PROFILE_DEFAULTS[profile_column])

    home_profiles = team_history.loc[team_history['side'] == 'home', [
        'date',
        'team',
        'opponent',
        'recent_goals_for',
        'recent_goals_against',
        'recent_total_goals',
        'recent_btts_rate',
    ]].rename(
        columns={
            'team': 'home_team',
            'opponent': 'away_team',
            'recent_goals_for': 'home_recent_goals_for',
            'recent_goals_against': 'home_recent_goals_against',
            'recent_total_goals': 'home_recent_total_goals',
            'recent_btts_rate': 'home_recent_btts_rate',
        }
    )
    away_profiles = team_history.loc[team_history['side'] == 'away', [
        'date',
        'team',
        'opponent',
        'recent_goals_for',
        'recent_goals_against',
        'recent_total_goals',
        'recent_btts_rate',
    ]].rename(
        columns={
            'team': 'away_team',
            'opponent': 'home_team',
            'recent_goals_for': 'away_recent_goals_for',
            'recent_goals_against': 'away_recent_goals_against',
            'recent_total_goals': 'away_recent_total_goals',
            'recent_btts_rate': 'away_recent_btts_rate',
        }
    )

    latest_profiles = (
        team_history.sort_values(['team', 'date', 'side'])
        .groupby('team')[['recent_goals_for', 'recent_goals_against', 'recent_total_goals', 'recent_btts_rate']]
        .last()
        .to_dict(orient='index')
    )
    return home_profiles, away_profiles, latest_profiles


def _build_goal_training_frame(matches_df, results_df):
    goal_history = matches_df.merge(
        results_df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'total_goals']],
        on=['date', 'home_team', 'away_team'],
        how='left',
    )
    goal_history = goal_history.dropna(subset=['home_score', 'away_score']).copy()
    goal_history['home_score'] = goal_history['home_score'].astype(int).clip(upper=EMPIRICAL_MAX_GOALS)
    goal_history['away_score'] = goal_history['away_score'].astype(int).clip(upper=EMPIRICAL_MAX_GOALS)
    goal_history['total_goals'] = goal_history['home_score'] + goal_history['away_score']
    goal_history['goal_diff'] = goal_history['home_score'] - goal_history['away_score']
    goal_history['result'] = goal_history.apply(
        lambda row: get_result_code(row['home_score'], row['away_score']),
        axis=1,
    )
    feature_frame = prepare_feature_frame(goal_history)
    probabilities = model.predict_proba(feature_frame)
    goal_history['p_home_raw'] = probabilities[:, 2]
    goal_history['p_draw_raw'] = probabilities[:, 1]
    goal_history['p_away_raw'] = probabilities[:, 0]

    adjusted_probabilities = goal_history.apply(
        lambda row: apply_modern_adjustment(
            p_home=row['p_home_raw'],
            p_draw=row['p_draw_raw'],
            p_away=row['p_away_raw'],
            team_a=row['home_team'],
            team_b=row['away_team'],
        )[:3],
        axis=1,
        result_type='expand',
    )
    adjusted_probabilities.columns = ['p_home', 'p_draw', 'p_away']
    goal_history[['p_home', 'p_draw', 'p_away']] = adjusted_probabilities

    home_profiles, away_profiles, latest_profiles = _build_team_goal_profiles(goal_history)
    goal_history = goal_history.merge(home_profiles, on=['date', 'home_team', 'away_team'], how='left')
    goal_history = goal_history.merge(away_profiles, on=['date', 'home_team', 'away_team'], how='left')
    goal_history.attrs['latest_team_goal_profiles'] = latest_profiles

    goal_history['abs_fifa_diff'] = goal_history['fifa_diff'].abs()
    goal_history['neutral'] = goal_history['neutral'].astype(float)
    goal_history['goal_env_home'] = goal_history['home_recent_goals_for'] + goal_history['away_recent_goals_against']
    goal_history['goal_env_away'] = goal_history['away_recent_goals_for'] + goal_history['home_recent_goals_against']
    goal_history['goal_env_total'] = goal_history['goal_env_home'] + goal_history['goal_env_away']
    return goal_history


def _build_goal_regression_features(frame):
    feature_frame = prepare_feature_frame(frame)
    feature_frame['p_home'] = frame['p_home'].to_numpy(dtype=float)
    feature_frame['p_draw'] = frame['p_draw'].to_numpy(dtype=float)
    feature_frame['p_away'] = frame['p_away'].to_numpy(dtype=float)
    feature_frame['abs_fifa_diff'] = frame['abs_fifa_diff'].to_numpy(dtype=float)
    for column in [
        'home_recent_goals_for',
        'home_recent_goals_against',
        'home_recent_total_goals',
        'home_recent_btts_rate',
        'away_recent_goals_for',
        'away_recent_goals_against',
        'away_recent_total_goals',
        'away_recent_btts_rate',
        'goal_env_home',
        'goal_env_away',
        'goal_env_total',
    ]:
        feature_frame[column] = frame[column].to_numpy(dtype=float)
    return feature_frame[GOAL_REGRESSION_FEATURE_COLUMNS].copy()


def _fit_goal_models(training_frame):
    regression_features = _build_goal_regression_features(training_frame)
    total_goal_model = GradientBoostingRegressor(
        random_state=GOAL_MODEL_RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=12,
        subsample=0.9,
    )
    goal_diff_model = GradientBoostingRegressor(
        random_state=GOAL_MODEL_RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=12,
        subsample=0.9,
    )
    total_goal_model.fit(regression_features, training_frame['total_goals'])
    goal_diff_model.fit(regression_features, training_frame['goal_diff'])
    return total_goal_model, goal_diff_model


def load_goal_models(home_model_path=GOAL_HOME_MODEL_PATH, away_model_path=GOAL_AWAY_MODEL_PATH):
    if not home_model_path.exists() or not away_model_path.exists():
        return None

    return joblib.load(home_model_path), joblib.load(away_model_path)


def train_and_save_goal_models(
    training_frame=None,
    home_model_path=GOAL_HOME_MODEL_PATH,
    away_model_path=GOAL_AWAY_MODEL_PATH,
):
    if training_frame is None:
        training_frame = _goal_training_frame_base

    total_goal_model, goal_diff_model = _fit_goal_models(training_frame)
    joblib.dump(total_goal_model, home_model_path)
    joblib.dump(goal_diff_model, away_model_path)
    return total_goal_model, goal_diff_model


def _attach_goal_model_predictions(training_frame, home_goal_model, away_goal_model):
    regression_features = _build_goal_regression_features(training_frame)
    enriched_frame = training_frame.copy()
    predicted_total_goals = np.clip(
        home_goal_model.predict(regression_features),
        0,
        EMPIRICAL_MAX_GOALS * 2,
    )
    predicted_goal_diff = np.clip(
        away_goal_model.predict(regression_features),
        -EMPIRICAL_MAX_GOALS,
        EMPIRICAL_MAX_GOALS,
    )
    enriched_frame['predicted_total_goals_model'] = predicted_total_goals
    enriched_frame['predicted_goal_diff_model'] = predicted_goal_diff
    enriched_frame['predicted_home_goals_model'] = np.clip(
        (predicted_total_goals + predicted_goal_diff) / 2,
        0,
        EMPIRICAL_MAX_GOALS,
    )
    enriched_frame['predicted_away_goals_model'] = np.clip(
        (predicted_total_goals - predicted_goal_diff) / 2,
        0,
        EMPIRICAL_MAX_GOALS,
    )
    return enriched_frame


def _build_match_feature_row(match_info):
    home_goal_profile = _latest_team_goal_profiles.get(match_info['team_a'], GOAL_PROFILE_DEFAULTS)
    away_goal_profile = _latest_team_goal_profiles.get(match_info['team_b'], GOAL_PROFILE_DEFAULTS)
    goal_env_home = float(home_goal_profile['recent_goals_for']) + float(away_goal_profile['recent_goals_against'])
    goal_env_away = float(away_goal_profile['recent_goals_for']) + float(home_goal_profile['recent_goals_against'])
    return {
        'fifa_diff': float(match_info['features']['fifa_diff_adjusted']),
        'abs_fifa_diff': abs(float(match_info['features']['fifa_diff_adjusted'])),
        'p_home': float(match_info['probabilities']['team_a_win']),
        'p_draw': float(match_info['probabilities']['draw']),
        'p_away': float(match_info['probabilities']['team_b_win']),
        'home_last5_win_rate': float(match_info['features']['form_a']),
        'away_last5_win_rate': float(match_info['features']['form_b']),
        'h2h_home_win_rate': float(match_info['features']['h2h_home_win_rate']),
        'h2h_draw_rate': float(match_info['features']['h2h_draw_rate']),
        'h2h_matches_played': float(match_info['features']['h2h_matches_played']),
        'home_penalty_win_rate': 0.5,
        'away_penalty_win_rate': 0.5,
        'neutral': float(match_info['features'].get('neutral', True)),
        'h2h_effective': float(match_info['features']['h2h_effective']),
        'home_recent_goals_for': float(match_info['features'].get('home_recent_goals_for', home_goal_profile['recent_goals_for'])),
        'home_recent_goals_against': float(match_info['features'].get('home_recent_goals_against', home_goal_profile['recent_goals_against'])),
        'home_recent_total_goals': float(match_info['features'].get('home_recent_total_goals', home_goal_profile['recent_total_goals'])),
        'home_recent_btts_rate': float(match_info['features'].get('home_recent_btts_rate', home_goal_profile['recent_btts_rate'])),
        'away_recent_goals_for': float(match_info['features'].get('away_recent_goals_for', away_goal_profile['recent_goals_for'])),
        'away_recent_goals_against': float(match_info['features'].get('away_recent_goals_against', away_goal_profile['recent_goals_against'])),
        'away_recent_total_goals': float(match_info['features'].get('away_recent_total_goals', away_goal_profile['recent_total_goals'])),
        'away_recent_btts_rate': float(match_info['features'].get('away_recent_btts_rate', away_goal_profile['recent_btts_rate'])),
        'goal_env_home': float(match_info['features'].get('goal_env_home', goal_env_home)),
        'goal_env_away': float(match_info['features'].get('goal_env_away', goal_env_away)),
        'goal_env_total': float(match_info['features'].get('goal_env_total', goal_env_home + goal_env_away)),
    }


def _poisson_probability(goal_count, rate, max_goal_count=EMPIRICAL_MAX_GOALS):
    safe_rate = max(float(rate), 1e-8)
    if goal_count < max_goal_count:
        return float(np.exp(goal_count * np.log(safe_rate) - safe_rate - lgamma(goal_count + 1)))

    capped_probability = 1.0
    for lower_goal_count in range(max_goal_count):
        capped_probability -= float(
            np.exp(lower_goal_count * np.log(safe_rate) - safe_rate - lgamma(lower_goal_count + 1))
        )
    return max(capped_probability, 0.0)


def _build_poisson_score_distribution(match_info, result=None, goal_models=None):
    predicted_home_goals, predicted_away_goals = _predict_goal_means(match_info, goal_models=goal_models)
    rows = []
    for home_score in range(EMPIRICAL_MAX_GOALS + 1):
        home_probability = _poisson_probability(home_score, predicted_home_goals)
        for away_score in range(EMPIRICAL_MAX_GOALS + 1):
            away_probability = _poisson_probability(away_score, predicted_away_goals)
            joint_probability = home_probability * away_probability
            rows.append({
                'home_score': home_score,
                'away_score': away_score,
                'weight': joint_probability,
            })

    distribution = pd.DataFrame(rows)
    if result is not None:
        distribution = distribution.loc[
            distribution.apply(
                lambda row: get_result_code(int(row['home_score']), int(row['away_score'])) == result,
                axis=1,
            )
        ].copy()
    if distribution.empty:
        return None

    weight_sum = float(distribution['weight'].sum())
    if weight_sum <= 0:
        return None

    distribution['probability'] = distribution['weight'] / weight_sum
    distribution = distribution.sort_values('probability', ascending=False).reset_index(drop=True)
    distribution['goal_diff'] = distribution['home_score'] - distribution['away_score']
    distribution['total_goals'] = distribution['home_score'] + distribution['away_score']
    return distribution[['home_score', 'away_score', 'weight', 'probability', 'goal_diff', 'total_goals']]


def _build_scoreline_distribution(match_info, result=None, goal_models=None):
    return _build_poisson_score_distribution(
        match_info,
        result=result,
        goal_models=goal_models,
    )


_goal_training_frame_base = _build_goal_training_frame(matches, historical_results)
_latest_team_goal_profiles = _goal_training_frame_base.attrs.get('latest_team_goal_profiles', {})
_loaded_goal_models = load_goal_models()
if _loaded_goal_models is None:
    _goal_home_model, _goal_away_model = _fit_goal_models(_goal_training_frame_base)
else:
    _goal_home_model, _goal_away_model = _loaded_goal_models

_goal_training_frame = _goal_training_frame_base.copy()
_goal_training_frame = _attach_goal_model_predictions(
    _goal_training_frame,
    _goal_home_model,
    _goal_away_model,
)


def build_match_info_from_row(row, fitted_model=model):
    feature_frame = prepare_feature_frame(pd.DataFrame([row]))
    probabilities = fitted_model.predict_proba(feature_frame)[0]
    p_away_raw, p_draw_raw, p_home_raw = probabilities
    p_home, p_draw, p_away, modern_diff, modern_boost = apply_modern_adjustment(
        p_home=float(p_home_raw),
        p_draw=float(p_draw_raw),
        p_away=float(p_away_raw),
        team_a=row['home_team'],
        team_b=row['away_team'],
    )

    return {
        'team_a': row['home_team'],
        'team_b': row['away_team'],
        'probabilities': {
            'team_a_win': p_home,
            'draw': p_draw,
            'team_b_win': p_away,
        },
        'raw_probabilities': {
            'team_a_win': float(p_home_raw),
            'draw': float(p_draw_raw),
            'team_b_win': float(p_away_raw),
        },
        'features': {
            'fifa_diff_raw': float(row['fifa_diff']),
            'fifa_diff_adjusted': float(row['fifa_diff']),
            'form_a': float(row['home_last5_win_rate']),
            'form_b': float(row['away_last5_win_rate']),
            'modern_diff': modern_diff,
            'modern_boost': modern_boost,
            'h2h_home_win_rate': float(row['h2h_home_win_rate']),
            'h2h_draw_rate': float(row['h2h_draw_rate']),
            'h2h_matches_played': float(row['h2h_matches_played']),
            'h2h_effective': float(row.get('h2h_effective', row['h2h_home_win_rate'] * min(1.0, row['h2h_matches_played'] / 10))),
            'neutral': bool(row['neutral']),
            'home_recent_goals_for': float(row.get('home_recent_goals_for', GOAL_PROFILE_DEFAULTS['recent_goals_for'])),
            'home_recent_goals_against': float(row.get('home_recent_goals_against', GOAL_PROFILE_DEFAULTS['recent_goals_against'])),
            'home_recent_total_goals': float(row.get('home_recent_total_goals', GOAL_PROFILE_DEFAULTS['recent_total_goals'])),
            'home_recent_btts_rate': float(row.get('home_recent_btts_rate', GOAL_PROFILE_DEFAULTS['recent_btts_rate'])),
            'away_recent_goals_for': float(row.get('away_recent_goals_for', GOAL_PROFILE_DEFAULTS['recent_goals_for'])),
            'away_recent_goals_against': float(row.get('away_recent_goals_against', GOAL_PROFILE_DEFAULTS['recent_goals_against'])),
            'away_recent_total_goals': float(row.get('away_recent_total_goals', GOAL_PROFILE_DEFAULTS['recent_total_goals'])),
            'away_recent_btts_rate': float(row.get('away_recent_btts_rate', GOAL_PROFILE_DEFAULTS['recent_btts_rate'])),
            'goal_env_home': float(row.get('goal_env_home', GOAL_PROFILE_DEFAULTS['recent_goals_for'] + GOAL_PROFILE_DEFAULTS['recent_goals_against'])),
            'goal_env_away': float(row.get('goal_env_away', GOAL_PROFILE_DEFAULTS['recent_goals_for'] + GOAL_PROFILE_DEFAULTS['recent_goals_against'])),
            'goal_env_total': float(row.get('goal_env_total', 2 * (GOAL_PROFILE_DEFAULTS['recent_goals_for'] + GOAL_PROFILE_DEFAULTS['recent_goals_against']))),
            'date': row['date'],
        },
    }


def _predict_goal_means(match_info, goal_models=None):
    total_goal_model, goal_diff_model = (
        goal_models if goal_models is not None else (_goal_home_model, _goal_away_model)
    )
    target_row = pd.DataFrame([_build_match_feature_row(match_info)])
    regression_features = _build_goal_regression_features(target_row)
    predicted_total_goals = float(np.clip(total_goal_model.predict(regression_features)[0], 0, EMPIRICAL_MAX_GOALS * 2))
    predicted_goal_diff = float(np.clip(goal_diff_model.predict(regression_features)[0], -EMPIRICAL_MAX_GOALS, EMPIRICAL_MAX_GOALS))
    predicted_home_goals = float(np.clip((predicted_total_goals + predicted_goal_diff) / 2, 0, EMPIRICAL_MAX_GOALS))
    predicted_away_goals = float(np.clip((predicted_total_goals - predicted_goal_diff) / 2, 0, EMPIRICAL_MAX_GOALS))
    return predicted_home_goals, predicted_away_goals


def estimate_expected_goals(match_info, goal_models=None):
    return _predict_goal_means(match_info, goal_models=goal_models)


def get_conditional_scoreline_distribution(match_info, result, goal_models=None):
    return _build_scoreline_distribution(
        match_info,
        result=result,
        goal_models=goal_models,
    )


def evaluate_second_stage_on_historical_matches(
    validation_start_date=DEFAULT_VALIDATION_START_DATE,
    max_matches=None,
    min_history_matches=1000,
):
    goal_history = _goal_training_frame.sort_values('date').reset_index(drop=True)
    training_frame = goal_history.loc[
        goal_history['date'] < pd.Timestamp(validation_start_date)
    ].copy()
    validation_frame = goal_history.loc[
        goal_history['date'] >= pd.Timestamp(validation_start_date)
    ].copy()

    if len(training_frame) < min_history_matches:
        return pd.DataFrame()

    validation_home_model, validation_away_model = _fit_goal_models(training_frame)

    if max_matches is not None:
        validation_frame = validation_frame.head(max_matches).copy()

    evaluation_rows = []
    for _, row in validation_frame.iterrows():
        match_info = build_match_info_from_row(row)
        predicted_goals_a, predicted_goals_b = estimate_expected_goals(
            match_info,
            goal_models=(validation_home_model, validation_away_model),
        )
        conditional_distribution = _build_scoreline_distribution(
            match_info,
            result=row['result'],
            goal_models=(validation_home_model, validation_away_model),
        )

        if conditional_distribution is None:
            continue

        actual_score_probability = conditional_distribution.loc[
            (conditional_distribution['home_score'] == row['home_score'])
            & (conditional_distribution['away_score'] == row['away_score']),
            'probability',
        ].sum()
        actual_score_probability = float(actual_score_probability)

        top_score = conditional_distribution.iloc[0]
        evaluation_rows.append({
            'date': row['date'],
            'team_a': row['home_team'],
            'team_b': row['away_team'],
            'actual_result': row['result'],
            'actual_goals_a': int(row['home_score']),
            'actual_goals_b': int(row['away_score']),
            'predicted_expected_goals_a': predicted_goals_a,
            'predicted_expected_goals_b': predicted_goals_b,
            'predicted_total_goals': predicted_goals_a + predicted_goals_b,
            'actual_total_goals': int(row['total_goals']),
            'predicted_goal_diff': predicted_goals_a - predicted_goals_b,
            'actual_goal_diff': int(row['goal_diff']),
            'conditional_top_score': f"{int(top_score['home_score'])}-{int(top_score['away_score'])}",
            'conditional_top_score_probability': float(top_score['probability']),
            'actual_score_probability': actual_score_probability,
            'exact_score_hit_at_top1': bool(
                int(top_score['home_score']) == int(row['home_score'])
                and int(top_score['away_score']) == int(row['away_score'])
            ),
            'goal_error_a': abs(predicted_goals_a - float(row['home_score'])),
            'goal_error_b': abs(predicted_goals_b - float(row['away_score'])),
            'total_goal_error': abs((predicted_goals_a + predicted_goals_b) - float(row['total_goals'])),
            'goal_diff_error': abs((predicted_goals_a - predicted_goals_b) - float(row['goal_diff'])),
            'actual_score_log_probability': float(np.log(max(actual_score_probability, 1e-12))),
        })

    return pd.DataFrame(evaluation_rows)


def summarize_second_stage_validation(validation_results):
    if validation_results.empty:
        return pd.DataFrame()

    summary = {
        'matches_evaluated': int(len(validation_results)),
        'home_goals_mae': float(validation_results['goal_error_a'].mean()),
        'away_goals_mae': float(validation_results['goal_error_b'].mean()),
        'total_goals_mae': float(validation_results['total_goal_error'].mean()),
        'goal_diff_mae': float(validation_results['goal_diff_error'].mean()),
        'home_goals_rmse': float(np.sqrt(np.mean((validation_results['predicted_expected_goals_a'] - validation_results['actual_goals_a']) ** 2))),
        'away_goals_rmse': float(np.sqrt(np.mean((validation_results['predicted_expected_goals_b'] - validation_results['actual_goals_b']) ** 2))),
        'total_goals_rmse': float(np.sqrt(np.mean((validation_results['predicted_total_goals'] - validation_results['actual_total_goals']) ** 2))),
        'goal_diff_rmse': float(np.sqrt(np.mean((validation_results['predicted_goal_diff'] - validation_results['actual_goal_diff']) ** 2))),
        'conditional_exact_score_top1_accuracy': float(validation_results['exact_score_hit_at_top1'].mean()),
        'mean_actual_score_probability': float(validation_results['actual_score_probability'].mean()),
        'mean_log_actual_score_probability': float(validation_results['actual_score_log_probability'].mean()),
        'mean_top1_score_probability': float(validation_results['conditional_top_score_probability'].mean()),
    }
    return pd.DataFrame([summary])


def sample_scoreline(match_info, result):
    distribution = _build_scoreline_distribution(
        match_info,
        result=result,
    )
    if distribution is None or distribution.empty:
        return None
    selected_index = np.random.choice(distribution.index.to_numpy(), p=distribution['probability'].to_numpy(dtype=float))
    selected_row = distribution.loc[selected_index]
    return int(selected_row['home_score']), int(selected_row['away_score'])
