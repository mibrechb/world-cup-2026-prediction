import numpy as np
import pandas as pd

from utils.runtime_data import (
    ROUND_VARIANCE,
    fifa_points,
    h2h_stats,
    historical_penalty,
    last5_win_rate,
    model,
    modern_strength,
    penalty_win_rate,
    smooth_form,
)

FEATURE_COLUMNS = [
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
]


def prepare_feature_frame(df):
    frame = df.copy()
    frame['fifa_diff_x_home_form'] = frame['fifa_diff'] * frame['home_last5_win_rate']
    frame['fifa_diff_x_away_form'] = frame['fifa_diff'] * frame['away_last5_win_rate']
    if 'h2h_effective' not in frame.columns:
        h2h_weight = np.minimum(1.0, frame['h2h_matches_played'] / 10)
        frame['h2h_effective'] = frame['h2h_home_win_rate'] * h2h_weight
    return frame[FEATURE_COLUMNS].copy()


def safe_normalize_probabilities(p_home, p_draw, p_away):
    probabilities = np.array([p_home, p_draw, p_away], dtype=float)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    probabilities /= total
    return tuple(probabilities)


def extract_model_probabilities(features, fitted_model=model):
    feature_frame = prepare_feature_frame(features)
    p_away, p_draw, p_home = fitted_model.predict_proba(feature_frame)[0]
    return float(p_home), float(p_draw), float(p_away)


def apply_modern_adjustment(p_home, p_draw, p_away, team_a, team_b):
    modern_diff = modern_strength(team_a) - modern_strength(team_b)
    modern_boost = 1 / (1 + np.exp(-modern_diff / 70))

    non_draw_probability = p_home + p_away
    adjusted_home = p_home * modern_boost
    adjusted_away = p_away * (1 - modern_boost)
    adjusted_non_draw = adjusted_home + adjusted_away
    if adjusted_non_draw > 0:
        rescale = non_draw_probability / adjusted_non_draw
        p_home = adjusted_home * rescale
        p_away = adjusted_away * rescale

    p_home, p_draw, p_away = safe_normalize_probabilities(p_home, p_draw, p_away)
    return p_home, p_draw, p_away, modern_diff, modern_boost


def sample_result_from_probabilities(p_home, p_draw, p_away):
    return np.random.choice(['A', 'D', 'B'], p=[p_home, p_draw, p_away])


def get_match_probabilities(team_a, team_b, fitted_model=model, phase='group', round_name=None):
    if round_name is not None:
        phase = round_name

    key = tuple(sorted([team_a, team_b]))
    h2h = h2h_stats.get(
        key,
        {'home_win_rate': 0.5, 'draw_rate': 0.0, 'matches_played': 1},
    )
    fifa_diff_raw = fifa_points.get(team_a, 1500) - fifa_points.get(team_b, 1500)
    form_a = smooth_form(last5_win_rate.get(team_a, 0.5))
    form_b = smooth_form(last5_win_rate.get(team_b, 0.5))
    round_variance = ROUND_VARIANCE.get(phase, ROUND_VARIANCE['group'])
    fifa_diff = fifa_diff_raw
    fifa_diff += np.random.normal(0, round_variance)
    fifa_diff += historical_penalty(team_a) - historical_penalty(team_b)
    h2h_weight = min(1.0, h2h['matches_played'] / 10)
    h2h_effective = h2h['home_win_rate'] * h2h_weight

    feature_row = pd.DataFrame([
        {
            'fifa_diff': fifa_diff,
            'home_last5_win_rate': form_a,
            'away_last5_win_rate': form_b,
            'h2h_home_win_rate': h2h['home_win_rate'],
            'h2h_draw_rate': h2h['draw_rate'],
            'h2h_matches_played': h2h['matches_played'],
            'home_penalty_win_rate': penalty_win_rate.get(team_a, 0.5),
            'away_penalty_win_rate': penalty_win_rate.get(team_b, 0.5),
            'neutral': True,
            'h2h_effective': h2h_effective,
        }
    ])

    p_home_raw, p_draw_raw, p_away_raw = extract_model_probabilities(feature_row, fitted_model=fitted_model)
    p_home, p_draw, p_away, modern_diff, modern_boost = apply_modern_adjustment(
        p_home=p_home_raw,
        p_draw=p_draw_raw,
        p_away=p_away_raw,
        team_a=team_a,
        team_b=team_b,
    )
    return {
        'team_a': team_a,
        'team_b': team_b,
        'probabilities': {
            'team_a_win': p_home,
            'draw': p_draw,
            'team_b_win': p_away,
        },
        'raw_probabilities': {
            'team_a_win': p_home_raw,
            'draw': p_draw_raw,
            'team_b_win': p_away_raw,
        },
        'features': {
            'fifa_diff_raw': fifa_diff_raw,
            'fifa_diff_adjusted': fifa_diff,
            'form_a': form_a,
            'form_b': form_b,
            'modern_diff': modern_diff,
            'modern_boost': modern_boost,
            'h2h_home_win_rate': h2h['home_win_rate'],
            'h2h_draw_rate': h2h['draw_rate'],
            'h2h_matches_played': h2h['matches_played'],
            'h2h_effective': h2h_effective,
            'neutral': True,
        },
    }
