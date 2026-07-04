import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from utils.outcome_model import get_match_probabilities, sample_result_from_probabilities
from utils.goal_model import estimate_expected_goals, sample_scoreline
from utils.runtime_data import groups, model

def _enrich_match_info_with_goal_estimate(match_info):
    expected_goals = estimate_expected_goals(match_info)
    if expected_goals is None:
        raise RuntimeError(
            'No historical goal expectation could be estimated for this match '
            'context from the second-stage model.'
        )

    team_a = match_info['team_a']
    team_b = match_info['team_b']
    match_info['expected_goals'] = {
        team_a: expected_goals[0],
        team_b: expected_goals[1],
    }
    return match_info


def _sample_scoreline(match_info, result):
    sampled_scoreline = sample_scoreline(match_info, result)
    if sampled_scoreline is None:
        raise RuntimeError(
            'No conditional scoreline distribution was available for the sampled '
            'match context. Scoreline simulation is configured to fail rather '
            'than silently fall back to a secondary method.'
        )

    goals_a, goals_b = sampled_scoreline
    return goals_a, goals_b, 'poisson'


def simulate_match(
    team_a,
    team_b,
    model=model,
    phase='group',
    round_name=None,
    return_only_result=False,
):
    if round_name is not None:
        phase = round_name

    match_info = get_match_probabilities(
        team_a=team_a,
        team_b=team_b,
        fitted_model=model,
        phase=phase,
    )
    match_info = _enrich_match_info_with_goal_estimate(match_info)

    p_home = match_info['probabilities']['team_a_win']
    p_draw = match_info['probabilities']['draw']
    p_away = match_info['probabilities']['team_b_win']
    result = sample_result_from_probabilities(
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
    )
    goals_a, goals_b, scoreline_method_used = _sample_scoreline(
        match_info=match_info,
        result=result,
    )

    if return_only_result:
        return result

    return {
        **match_info,
        'result': result,
        'winner': team_a if result == 'A' else team_b if result == 'B' else None,
        'goals_a': goals_a,
        'goals_b': goals_b,
        'score': f'{goals_a}-{goals_b}',
        'scoreline_method': 'poisson',
        'scoreline_method_used': scoreline_method_used,
    }


def simulate_match_many(
    team_a,
    team_b,
    model=model,
    phase='group',
    round_name=None,
    n_simulations=10000,
    use_tqdm=True,
):
    if round_name is not None:
        phase = round_name

    rows = []
    for _ in tqdm(
        range(n_simulations),
        desc=f'Simulating {team_a} vs {team_b}',
        total=n_simulations,
        unit='sim',
        disable=(not use_tqdm),
    ):
        match = simulate_match(
            team_a=team_a,
            team_b=team_b,
            model=model,
            phase=phase,
            return_only_result=False,
        )
        rows.append({
            'team_a': team_a,
            'team_b': team_b,
            'result': match['result'],
            'winner': match['winner'],
            'goals_a': match['goals_a'],
            'goals_b': match['goals_b'],
            'score': match['score'],
            'scoreline_method': match['scoreline_method'],
            'scoreline_method_used': match['scoreline_method_used'],
            'team_a_win_probability': match['probabilities']['team_a_win'],
            'draw_probability': match['probabilities']['draw'],
            'team_b_win_probability': match['probabilities']['team_b_win'],
            'expected_goals_a': match['expected_goals'][team_a],
            'expected_goals_b': match['expected_goals'][team_b],
        })

    return pd.DataFrame(rows)


def get_result_from_goals(goals_a, goals_b):
    if goals_a > goals_b:
        return 'A'
    if goals_b > goals_a:
        return 'B'
    return 'D'


def score_tip(
    tip_goals_a,
    tip_goals_b,
    actual_goals_a,
    actual_goals_b,
    phase='group',
):
    if phase == 'group':
        winner_points = 5
        home_goals_points = 1
        away_goals_points = 1
        goal_difference_points = 3
    elif phase in {
        'play-in',
        'round_of_16',
        'round_of_32',
        'quarterfinal',
        'semifinal',
        'final',
    }:
        winner_points = 10
        home_goals_points = 2
        away_goals_points = 2
        goal_difference_points = 6
    else:
        raise ValueError(f'Unknown phase: {phase}')

    points = 0
    tip_result = get_result_from_goals(tip_goals_a, tip_goals_b)
    actual_result = get_result_from_goals(actual_goals_a, actual_goals_b)
    correct_winner = tip_result == actual_result
    if correct_winner:
        points += winner_points
    if tip_goals_a == actual_goals_a:
        points += home_goals_points
    if tip_goals_b == actual_goals_b:
        points += away_goals_points

    correct_goal_difference = (
        tip_goals_a - tip_goals_b == actual_goals_a - actual_goals_b
    )
    if correct_winner and correct_goal_difference:
        points += goal_difference_points
    return points


def find_optimal_tip_from_simulations(
    df_simulations,
    team_a,
    team_b,
    phase='group',
    max_tip_goals=6,
):
    rows = []
    for tip_goals_a in range(max_tip_goals + 1):
        for tip_goals_b in range(max_tip_goals + 1):
            points = df_simulations.apply(
                lambda row: score_tip(
                    tip_goals_a=tip_goals_a,
                    tip_goals_b=tip_goals_b,
                    actual_goals_a=row['goals_a'],
                    actual_goals_b=row['goals_b'],
                    phase=phase,
                ),
                axis=1,
            )
            tip_result = get_result_from_goals(tip_goals_a, tip_goals_b)
            if tip_result == 'A':
                outcome = f'{team_a} win'
            elif tip_result == 'B':
                outcome = f'{team_b} win'
            else:
                outcome = 'Draw'

            exact_score_probability = (
                (
                    (df_simulations['goals_a'] == tip_goals_a)
                    & (df_simulations['goals_b'] == tip_goals_b)
                ).mean()
                * 100
            )
            rows.append({
                'tip': f'{tip_goals_a}-{tip_goals_b}',
                'tip_goals_a': tip_goals_a,
                'tip_goals_b': tip_goals_b,
                'outcome': outcome,
                'expected_points': points.mean(),
                'median_points': points.median(),
                'max_points': points.max(),
                'exact_score_probability_percent': exact_score_probability,
            })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ['expected_points', 'exact_score_probability_percent'],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def reverse_score(score):
    goals_a, goals_b = score.split('-')
    return f'{goals_b}-{goals_a}'


def mirror_outcome_label(outcome, team_a, team_b):
    return (
        outcome
        .replace(team_a, '__TEMP__')
        .replace(team_b, team_a)
        .replace('__TEMP__', team_b)
    )


def build_matchup_detail_tables(df_simulations, df_tiprank, team_a, team_b):
    outcome_label_map = {
        'A': f'{team_a} win',
        'D': 'Draw',
        'B': f'{team_b} win',
    }
    score_distribution = (
        df_simulations['score']
        .value_counts(normalize=True)
        .mul(100)
        .rename_axis('score')
        .reset_index(name='probability_percent')
    )
    score_distribution['team_a'] = team_a
    score_distribution['team_b'] = team_b
    score_distribution['rank'] = np.arange(1, len(score_distribution) + 1)

    outcome_distribution = (
        df_simulations['result']
        .map(outcome_label_map)
        .value_counts(normalize=True)
        .reindex([f'{team_a} win', 'Draw', f'{team_b} win'], fill_value=0)
        .mul(100)
        .rename_axis('outcome')
        .reset_index(name='probability_percent')
    )
    outcome_distribution['team_a'] = team_a
    outcome_distribution['team_b'] = team_b
    outcome_distribution['rank'] = np.arange(1, len(outcome_distribution) + 1)

    goal_diff_distribution = (
        (df_simulations['goals_a'] - df_simulations['goals_b'])
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .rename_axis('goal_diff')
        .reset_index(name='probability_percent')
    )
    goal_diff_distribution['team_a'] = team_a
    goal_diff_distribution['team_b'] = team_b
    goal_diff_distribution['rank'] = np.arange(1, len(goal_diff_distribution) + 1)

    tip_rank = df_tiprank.copy().reset_index(drop=True)
    tip_rank['team_a'] = team_a
    tip_rank['team_b'] = team_b
    tip_rank['rank'] = np.arange(1, len(tip_rank) + 1)

    summary = {
        'team_a': team_a,
        'team_b': team_b,
        'team_a_win_probability': float(
            outcome_distribution.loc[
                outcome_distribution['outcome'] == f'{team_a} win',
                'probability_percent',
            ].iloc[0]
        ),
        'draw_probability': float(
            outcome_distribution.loc[
                outcome_distribution['outcome'] == 'Draw',
                'probability_percent',
            ].iloc[0]
        ),
        'team_b_win_probability': float(
            outcome_distribution.loc[
                outcome_distribution['outcome'] == f'{team_b} win',
                'probability_percent',
            ].iloc[0]
        ),
        'most_common_score': str(score_distribution.iloc[0]['score']),
        'most_common_score_probability': float(
            score_distribution.iloc[0]['probability_percent']
        ),
        'recommended_tip': str(tip_rank.iloc[0]['tip']),
        'recommended_tip_outcome': str(tip_rank.iloc[0]['outcome']),
        'recommended_tip_expected_points': float(
            tip_rank.iloc[0]['expected_points']
        ),
        'recommended_tip_exact_probability': float(
            tip_rank.iloc[0]['exact_score_probability_percent']
        ),
        'avg_goals_a': float(df_simulations['goals_a'].mean()),
        'avg_goals_b': float(df_simulations['goals_b'].mean()),
        'avg_goal_difference': float(
            (df_simulations['goals_a'] - df_simulations['goals_b']).mean()
        ),
    }

    return {
        'summary': summary,
        'score_distribution': score_distribution,
        'outcome_distribution': outcome_distribution,
        'goal_diff_distribution': goal_diff_distribution,
        'tip_rank': tip_rank,
    }


def mirror_matchup_detail_tables(matchup_tables):
    team_a = matchup_tables['summary']['team_a']
    team_b = matchup_tables['summary']['team_b']

    mirrored_summary = matchup_tables['summary'].copy()
    mirrored_summary.update({
        'team_a': team_b,
        'team_b': team_a,
        'team_a_win_probability': matchup_tables['summary']['team_b_win_probability'],
        'team_b_win_probability': matchup_tables['summary']['team_a_win_probability'],
        'draw_probability': matchup_tables['summary']['draw_probability'],
        'most_common_score': reverse_score(matchup_tables['summary']['most_common_score']),
        'recommended_tip': reverse_score(matchup_tables['summary']['recommended_tip']),
        'recommended_tip_outcome': mirror_outcome_label(
            matchup_tables['summary']['recommended_tip_outcome'],
            team_a=team_a,
            team_b=team_b,
        ),
        'avg_goals_a': matchup_tables['summary']['avg_goals_b'],
        'avg_goals_b': matchup_tables['summary']['avg_goals_a'],
        'avg_goal_difference': -matchup_tables['summary']['avg_goal_difference'],
    })

    mirrored_score_distribution = matchup_tables['score_distribution'].copy()
    mirrored_score_distribution['team_a'] = team_b
    mirrored_score_distribution['team_b'] = team_a
    mirrored_score_distribution['score'] = mirrored_score_distribution['score'].map(reverse_score)

    mirrored_outcome_distribution = matchup_tables['outcome_distribution'].copy()
    mirrored_outcome_distribution['team_a'] = team_b
    mirrored_outcome_distribution['team_b'] = team_a
    outcome_order = [f'{team_b} win', 'Draw', f'{team_a} win']
    mirrored_outcome_distribution['sort_order'] = mirrored_outcome_distribution['outcome'].map(
        {label: index for index, label in enumerate(outcome_order)}
    )
    mirrored_outcome_distribution = (
        mirrored_outcome_distribution
        .sort_values('sort_order')
        .drop(columns='sort_order')
        .reset_index(drop=True)
    )
    mirrored_outcome_distribution['rank'] = np.arange(1, len(mirrored_outcome_distribution) + 1)

    mirrored_goal_diff_distribution = matchup_tables['goal_diff_distribution'].copy()
    mirrored_goal_diff_distribution['team_a'] = team_b
    mirrored_goal_diff_distribution['team_b'] = team_a
    mirrored_goal_diff_distribution['goal_diff'] = -mirrored_goal_diff_distribution['goal_diff']
    mirrored_goal_diff_distribution = mirrored_goal_diff_distribution.sort_values('goal_diff').reset_index(drop=True)
    mirrored_goal_diff_distribution['rank'] = np.arange(1, len(mirrored_goal_diff_distribution) + 1)

    mirrored_tip_rank = matchup_tables['tip_rank'].copy()
    mirrored_tip_rank['team_a'] = team_b
    mirrored_tip_rank['team_b'] = team_a
    mirrored_tip_rank['tip'] = mirrored_tip_rank['tip'].map(reverse_score)
    mirrored_tip_rank['tip_goals_a'], mirrored_tip_rank['tip_goals_b'] = zip(
        *mirrored_tip_rank['tip'].map(lambda score: tuple(map(int, score.split('-'))))
    )
    mirrored_tip_rank['outcome'] = mirrored_tip_rank['outcome'].map(
        lambda outcome: mirror_outcome_label(outcome, team_a=team_a, team_b=team_b)
    )

    return {
        'summary': mirrored_summary,
        'score_distribution': mirrored_score_distribution,
        'outcome_distribution': mirrored_outcome_distribution,
        'goal_diff_distribution': mirrored_goal_diff_distribution,
        'tip_rank': mirrored_tip_rank,
    }
