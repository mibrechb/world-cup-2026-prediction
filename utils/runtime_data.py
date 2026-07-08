from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd

CURRENT_YEAR = 2026
EMPIRICAL_MAX_GOALS = 7
DEFAULT_PHASE = os.environ.get('WORLD_CUP_PHASE', 'round_of_32')
ROUND_VARIANCE = {
    'group': 18,
    'round_of_32': 30,
    'round_of_16': 45,
    'quarterfinal': 55,
    'semifinal': 70,
    'third_place': 70,
    'final': 85,
}
PLAYOFF_REPLACEMENTS = {
    'UEFA_Playoff_A': 'Bosnia and Herzegovina',
    'UEFA_Playoff_B': 'Sweden',
    'UEFA_Playoff_C': 'Turkey',
    'UEFA_Playoff_D': 'Czech Republic',
    'FIFA_Playoff_1': 'DR Congo',
    'FIFA_Playoff_2': 'Iraq',
}
GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'UEFA_Playoff_D'],
    'B': ['Canada', 'Switzerland', 'Qatar', 'UEFA_Playoff_A'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'UEFA_Playoff_C'],
    'E': ['Germany', 'Ivory Coast', 'Ecuador', 'Curaçao'],
    'F': ['Netherlands', 'Japan', 'Tunisia', 'UEFA_Playoff_B'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Uruguay', 'Saudi Arabia', 'Cape Verde'],
    'I': ['France', 'Senegal', 'Norway', 'FIFA_Playoff_2'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Colombia', 'Uzbekistan', 'FIFA_Playoff_1'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}
groups = {
    group: [PLAYOFF_REPLACEMENTS.get(team, team) for team in teams]
    for group, teams in GROUPS.items()
}

# Define teams of other phases
TEAMS_ROUND_OF_32 = [
    'South Africa', 'Canada', 'Netherlands', 'Morocco',
    'Germany', 'Paraguay', 'France', 'Sweden',
    'Brazil', 'Japan', 'Norway', 'Ivory Coast',
    'Mexico', 'Ecuador', 'England', 'DR Congo',
    'Portugal', 'Croatia', 'Spain', 'Austria',
    'United States', 'Bosnia and Herzegovina', 'Belgium', 'Senegal',
    'Argentina', 'Cape Verde', 'Egypt', 'Australia',
    'Switzerland', 'Algeria', 'Colombia', 'Ghana',
]

TEAMS_ROUND_OF_16 = [
    'Canada', 'Morocco', 'Paraguay', 'France',
    'Brazil', 'Norway', 'Mexico', 'England',
    'Portugal', 'Spain', 'United States', 'Belgium',
    'Argentina', 'Egypt', 'Switzerland', 'Colombia',
]

TEAMS_QUARTERFINAL = [
    'Morocco', 'France',
    'Norway', 'England',
    'Spain', 'Belgium',
    'Argentina', 'Switzerland',
]

# Load phase-specific outcome model
def get_outcome_model_path(phase=DEFAULT_PHASE, model_dir='models'):
    """Returns the preferred phase-specific outcome model path."""
    return Path(model_dir) / f'model_grboostclass_outcome_{phase}.pkl'

def load_outcome_model(phase=DEFAULT_PHASE, model_dir='models'):
    """Loads the required phase-specific outcome model."""
    phase_model_path = get_outcome_model_path(phase, model_dir=model_dir)
    if not phase_model_path.exists():
        raise FileNotFoundError(
            f'Missing phase-specific outcome model: {phase_model_path}'
        )
    return joblib.load(phase_model_path)

def configure_outcome_model(phase=DEFAULT_PHASE):
    """Loads and stores the active outcome model for a phase."""
    global model
    model = load_outcome_model(phase)
    print(f"Loaded outcome model for phase '{phase}' from {get_outcome_model_path(phase)}")
    return model

model = configure_outcome_model(DEFAULT_PHASE)

# Phase-specific fifa rankings (latest)
PHASE_KNOWLEDGE_DATES = {
    'group': '2026-06-11',
    'round_of_32': '2026-06-28',
    'round_of_16': '2026-07-04',
    'quarterfinal': '2026-07-09',
    'semifinal': '2026-07-14',
    'final': '2026-07-19',
}

def load_fifa_ranking_snapshot(
    phase: str = DEFAULT_PHASE,
    ranking_path: str = 'data/processed/fifa_world_rankings.csv',
) -> pd.DataFrame:
    """Loads the latest FIFA ranking snapshot available by phase start date."""
    df_fifa_all = pd.read_csv(ranking_path, parse_dates=['date'])
    df_fifa_all['date'] = pd.to_datetime(df_fifa_all['date']).dt.normalize()

    phase_cutoff_date = pd.Timestamp(PHASE_KNOWLEDGE_DATES[phase])

    df_available_dates = (
        df_fifa_all.loc[df_fifa_all['date'] <= phase_cutoff_date, ['date']]
        .drop_duplicates()
        .sort_values('date')
    )

    if df_available_dates.empty:
        raise ValueError(
            f"No FIFA ranking snapshot available by {phase_cutoff_date.date()} "
            f"for phase '{phase}'."
        )

    ranking_snapshot_date = df_available_dates['date'].iloc[-1]

    print(
        f"Loaded FIFA rankings from {ranking_snapshot_date.date()} "
        f"for phase '{phase}'"
    )

    # Rename countries to align
    dict_rename = {'Türkiye': 'Turkey', 
                   "Côte d'Ivoire": 'Ivory Coast', 
                   'Czechia': 'Czech Republic',
                   'DPR Korea': 'North Korea',
                   'Congo DR': 'DR Congo',
                   }
    df_fifa_all.replace({'country': dict_rename}, inplace=True)

    return df_fifa_all[df_fifa_all['date'] == ranking_snapshot_date].copy()

fifa = load_fifa_ranking_snapshot(DEFAULT_PHASE)

# Other historical information
matches = pd.read_csv('data/processed/matches_2000_onwards_features_fifa.csv', parse_dates=['date'])
historical_results = pd.read_csv('data/processed/results_2000_onwards.csv', parse_dates=['date'])
shootouts = pd.read_csv('data/raw/shootouts.csv', parse_dates=['date'])
achievements = pd.read_csv('data/raw/team_achievements.csv').set_index('team')

def achievement_score(team):
    if team not in achievements.index:
        return 0.0

    row = achievements.loc[team]
    score = 0.0
    if not np.isnan(row['wc_last_semi_year']):
        score += 0.6 * np.exp(-(CURRENT_YEAR - row['wc_last_semi_year']) / 6)
    if not np.isnan(row['wc_last_final_year']):
        score += 0.9 * np.exp(-(CURRENT_YEAR - row['wc_last_final_year']) / 6)
    if not np.isnan(row['wc_last_win_year']):
        score += 1.0 * np.exp(-(CURRENT_YEAR - row['wc_last_win_year']) / 8)
    if not np.isnan(row['cont_last_semi_year']):
        score += 0.4 * np.exp(-(CURRENT_YEAR - row['cont_last_semi_year']) / 5)
    if not np.isnan(row['cont_last_final_year']):
        score += 0.6 * np.exp(-(CURRENT_YEAR - row['cont_last_final_year']) / 5)
    if not np.isnan(row['cont_last_win_year']):
        score += 0.8 * np.exp(-(CURRENT_YEAR - row['cont_last_win_year']) / 6)
    return score


def historical_penalty(team):
    if team not in achievements.index:
        return 0.0

    row = achievements.loc[team]
    if (
        np.isnan(row['wc_last_semi_year'])
        and np.isnan(row['cont_last_semi_year'])
        and (CURRENT_YEAR - row['last_world_cup_participation'] > 20)
    ):
        return -5
    return 0.0


def smooth_form(win_rate, alpha=0.65):
    return alpha * win_rate + (1 - alpha) * 0.5


def modern_strength(team):
    score = (smooth_form(last5_win_rate.get(team, 0.5)) - 0.5) * 110
    if team not in achievements.index:
        return score

    row = achievements.loc[team]
    if not np.isnan(row['wc_last_semi_year']):
        score += 22 * np.exp(-(CURRENT_YEAR - row['wc_last_semi_year']) / 6)
    if not np.isnan(row['wc_last_final_year']):
        score += 28 * np.exp(-(CURRENT_YEAR - row['wc_last_final_year']) / 6)
    if not np.isnan(row['wc_last_win_year']):
        score += 32 * np.exp(-(CURRENT_YEAR - row['wc_last_win_year']) / 8)
    if not np.isnan(row['cont_last_semi_year']):
        score += 14 * np.exp(-(CURRENT_YEAR - row['cont_last_semi_year']) / 5)
    if not np.isnan(row['cont_last_final_year']):
        score += 18 * np.exp(-(CURRENT_YEAR - row['cont_last_final_year']) / 5)
    if not np.isnan(row['cont_last_win_year']):
        score += 22 * np.exp(-(CURRENT_YEAR - row['cont_last_win_year']) / 6)

    modern_teams = {
        'Argentina': 6,
        'Spain': 6,
        'France': 6,
        'England': 5,
        'Morocco': 5,
        'Portugal': 5,
        'Germany': 4,
        'Brazil': 4,
        'Belgium': 4,
        'Netherlands': 4,
        'Croatia': 4,
        'Japan': 3,
        'Senegal': 3,
        'Uruguay': 3,
    }
    return score + modern_teams.get(team, 0)


def get_result_code(goals_a, goals_b):
    if goals_a > goals_b:
        return 'A'
    if goals_b > goals_a:
        return 'B'
    return 'D'


fifa_points = dict(zip(fifa['country'], fifa['total_points']))
penalty_wins = shootouts['winner'].value_counts()
penalty_games = pd.concat([shootouts['home_team'], shootouts['away_team']]).value_counts()
penalty_win_rate = (penalty_wins / penalty_games).fillna(0.5)

last5_win_rate = {}
for team, group in matches.groupby('home_team'):
    last5_win_rate[team] = group['home_last5_win_rate'].iloc[-1] if len(group) else 0.5
for team, group in matches.groupby('away_team'):
    last5_win_rate[team] = group['away_last5_win_rate'].iloc[-1] if len(group) else 0.5

h2h_stats = {}
for _, row in matches.iterrows():
    key = tuple(sorted([row['home_team'], row['away_team']]))
    h2h_stats[key] = {
        'home_win_rate': row['h2h_home_win_rate'],
        'draw_rate': row['h2h_draw_rate'],
        'matches_played': max(1, row['h2h_matches_played']),
    }

playoff_candidates = {
    'UEFA_Playoff_A': ['Italy', 'Bosnia and Herzegovina', 'Northern Ireland', 'Wales'],
    'UEFA_Playoff_B': ['Albania', 'Poland', 'Sweden', 'Ukraine'],
    'UEFA_Playoff_C': ['Kosovo', 'Romania', 'Slovakia', 'Turkey'],
    'UEFA_Playoff_D': ['Denmark', 'Republic of Ireland', 'North Macedonia', 'Czech Republic'],
    'FIFA_Playoff_1': ['Jamaica', 'New Caledonia', 'DR Congo'],
    'FIFA_Playoff_2': ['Bolivia', 'Iraq', 'Suriname'],
}
for playoff, teams in playoff_candidates.items():
    fifa_points[playoff] = np.mean([fifa_points[t] for t in teams])
    last5_win_rate[playoff] = np.mean([last5_win_rate.get(t, 0.5) for t in teams])
