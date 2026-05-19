import pandas as pd
import numpy as np
import joblib
from tqdm.notebook import tqdm

CURRENT_YEAR = 2026

# Standard deviation of the random noise added to the FIFA-point difference
# before each simulated match. Higher values make the match outcome more
# volatile. Later rounds use larger values to model knockout-stage uncertainty
# and tournament chaos.
ROUND_VARIANCE = {
    "group": 18,
    "play-in": 30,
    "round_of_16": 45,
    "quarterfinal": 55,
    "semifinal": 70,
    "final": 85
}

# Expected avg. goals by round
# used as lambda parameter for the Poisson distribution 
# to simulate match scores
GOALS_BY_ROUND = {
    'group': 2.55,
    'play-in': 2.40,
    'round_of_16': 2.30,
    'quarterfinal': 2.20,
    'semifinal': 2.10,
    'final': 2.00,
}

# Groups
PLAYOFF_REPLACEMENTS = {
    'UEFA_Playoff_A': 'Bosnia and Herzegovina',
    'UEFA_Playoff_B': 'Sweden',
    'UEFA_Playoff_C': 'Turkey',
    'UEFA_Playoff_D': 'Czech Republic',
    'FIFA_Playoff_1': 'DR Congo',
    'FIFA_Playoff_2': 'Iraq',
}

GROUPS = {
    "A":["Mexico","South Africa","South Korea","UEFA_Playoff_D"],
    "B":["Canada","Switzerland","Qatar","UEFA_Playoff_A"],
    "C":["Brazil","Morocco","Haiti","Scotland"],
    "D":["United States","Paraguay","Australia","UEFA_Playoff_C"],
    "E":["Germany","Ivory Coast","Ecuador","Curaçao"],
    "F":["Netherlands","Japan","Tunisia","UEFA_Playoff_B"],
    "G":["Belgium","Egypt","Iran","New Zealand"],
    "H":["Spain","Uruguay","Saudi Arabia","Cape Verde"],
    "I":["France","Senegal","Norway","FIFA_Playoff_2"],
    "J":["Argentina","Algeria","Austria","Jordan"],
    "K":["Portugal","Colombia","Uzbekistan","FIFA_Playoff_1"],
    "L":["England","Croatia","Ghana","Panama"]
}

groups = {
    group: [PLAYOFF_REPLACEMENTS.get(team, team) for team in teams]
    for group, teams in GROUPS.items()
}

# load data and model
model = joblib.load("models/gradient_boosting_v1.pkl")
matches = pd.read_csv("data/processed/matches_2000_onwards_features_fifa.csv", parse_dates=["date"])
shootouts = pd.read_csv("data/raw/shootouts.csv", parse_dates=["date"])
fifa = pd.read_csv("data/processed/fifa_latest_world_ranking.csv", parse_dates=["date"])
achievements = pd.read_csv("data/raw/team_achievements.csv").set_index("team")

# Utility functions
def achievement_score(team):
    if team not in achievements.index:
        return 0.0

    row = achievements.loc[team]
    score = 0.0

    # world cup (max ~2.5)
    if not np.isnan(row["wc_last_semi_year"]):
        wc_semi_recency = CURRENT_YEAR - row["wc_last_semi_year"]
        score += 0.6 * np.exp(-wc_semi_recency / 6)

    if not np.isnan(row["wc_last_final_year"]):
        wc_final_recency = CURRENT_YEAR - row["wc_last_final_year"]
        score += 0.9 * np.exp(-wc_final_recency / 6)

    if not np.isnan(row["wc_last_win_year"]):
        wc_win_recency = CURRENT_YEAR - row["wc_last_win_year"]
        score += 1.0 * np.exp(-wc_win_recency / 8)

    # continental (max ~1.8)
    if not np.isnan(row["cont_last_semi_year"]):
        cont_semi_recency = CURRENT_YEAR - row["cont_last_semi_year"]
        score += 0.4 * np.exp(-cont_semi_recency / 5)

    if not np.isnan(row["cont_last_final_year"]):
        cont_final_recency = CURRENT_YEAR - row["cont_last_final_year"]
        score += 0.6 * np.exp(-cont_final_recency / 5)

    if not np.isnan(row["cont_last_win_year"]):
        cont_win_recency = CURRENT_YEAR - row["cont_last_win_year"]
        score += 0.8 * np.exp(-cont_win_recency / 6)

    return score


def historical_penalty(team):
    if team not in achievements.index:
        return 0.0

    row = achievements.loc[team]

    # If the team has never reached the semifinals of any tournament and 
    # has been absent from the World Cup for over 20 years
    if (
        np.isnan(row["wc_last_semi_year"]) and
        np.isnan(row["cont_last_semi_year"]) and
        (CURRENT_YEAR - row["last_world_cup_participation"] > 20)
    ):
        return -5

    return 0.0


def smooth_form(win_rate, alpha=0.65):
    return alpha*win_rate + (1-alpha)*0.5

def modern_strength(team):
    score = 0.0

    # 1. RECENT FORM (base)
    score += (smooth_form(last5_win_rate.get(team, 0.5)) - 0.5) * 110

    if team not in achievements.index:
        return score

    row = achievements.loc[team]

    # 2. Modern World Cup
    if not np.isnan(row["wc_last_semi_year"]):
        years = CURRENT_YEAR - row["wc_last_semi_year"]
        score += 22 * np.exp(-years / 6)

    if not np.isnan(row["wc_last_final_year"]):
        years = CURRENT_YEAR - row["wc_last_final_year"]
        score += 28 * np.exp(-years / 6)

    if not np.isnan(row["wc_last_win_year"]):
        years = CURRENT_YEAR - row["wc_last_win_year"]
        score += 32 * np.exp(-years / 8)

    # 3. Modern Continental
    if not np.isnan(row["cont_last_semi_year"]):
        years = CURRENT_YEAR - row["cont_last_semi_year"]
        score += 14 * np.exp(-years / 5)

    if not np.isnan(row["cont_last_final_year"]):
        years = CURRENT_YEAR - row["cont_last_final_year"]
        score += 18 * np.exp(-years / 5)

    if not np.isnan(row["cont_last_win_year"]):
        years = CURRENT_YEAR - row["cont_last_win_year"]
        score += 22 * np.exp(-years / 6)

    # 4. Micro adjustment (optional)
    MODERN_TEAMS = {
        "Argentina": 6,
        "Spain": 6,
        "France": 6,
        "England": 5,
        "Morocco": 5,
        "Portugal": 5,
        "Germany": 4,
        "Brazil": 4,
        "Belgium": 4,
        "Netherlands": 4,
        "Croatia": 4,
        "Japan": 3,
        "Senegal": 3,
        "Uruguay": 3
    }

    score += MODERN_TEAMS.get(team, 0)

    return score


# FIFA points
fifa_points = dict(zip(fifa["country"], fifa["total_points"]))

# Penalty win rate
penalty_wins = shootouts["winner"].value_counts()
penalty_games = pd.concat([shootouts["home_team"], shootouts["away_team"]]).value_counts()
penalty_win_rate = (penalty_wins / penalty_games).fillna(0.5)

# Last5 form
last5_win_rate = {}
for team, group in matches.groupby("home_team"):
    last5_win_rate[team] = group["home_last5_win_rate"].iloc[-1] if len(group) else 0.5
for team, group in matches.groupby("away_team"):
    last5_win_rate[team] = group["away_last5_win_rate"].iloc[-1] if len(group) else 0.5

# H2H
h2h_stats = {}
for idx, row in matches.iterrows():
    key = tuple(sorted([row["home_team"], row["away_team"]]))
    h2h_stats[key] = {"home_win_rate": row["h2h_home_win_rate"], "draw_rate": row["h2h_draw_rate"], "matches_played": max(1,row["h2h_matches_played"])}

# PLAYOFF TEAMS
playoff_candidates = {
    "UEFA_Playoff_A":["Italy","Bosnia and Herzegovina","Northern Ireland","Wales"],
    "UEFA_Playoff_B":["Albania","Poland","Sweden","Ukraine"],
    "UEFA_Playoff_C":["Kosovo","Romania","Slovakia","Turkey"],
    "UEFA_Playoff_D":["Denmark","Republic of Ireland","North Macedonia","Czech Republic"],
    "FIFA_Playoff_1":["Jamaica","New Caledonia","DR Congo"],
    "FIFA_Playoff_2":["Bolivia","Iraq","Suriname"]
}
for playoff, teams in playoff_candidates.items():
    fifa_points[playoff] = np.mean([fifa_points[t] for t in teams])
    last5_win_rate[playoff] = np.mean([last5_win_rate.get(t,0.5) for t in teams])

def _safe_normalize_probabilities(p_home, p_draw, p_away):
    """Normalize match probabilities safely.

    Args:
        p_home: Probability that team A wins.
        p_draw: Probability of a draw.
        p_away: Probability that team B wins.

    Returns:
        Tuple of normalized probabilities: (p_home, p_draw, p_away).
    """
    probabilities = np.array([p_home, p_draw, p_away], dtype=float)
    probabilities = np.clip(probabilities, 0.0, None)

    total = probabilities.sum()
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3

    probabilities /= total
    return tuple(probabilities)

def _extract_model_probabilities(features, model=model):
    """Extract team A win, draw, and team B win probabilities.

    This follows the original project assumption:
        p_away, p_draw, p_home = model.predict_proba(features)[0]

    Args:
        model: Trained classifier with predict_proba.
        features: Single-row feature DataFrame.

    Returns:
        Tuple: (p_home, p_draw, p_away).
    """
    p_away, p_draw, p_home = model.predict_proba(features)[0]
    return float(p_home), float(p_draw), float(p_away)

def _apply_modern_adjustment(p_home, p_draw, p_away, team_a, team_b):
    """Apply the original modern-football adjustment.

    Args:
        p_home: Probability that team A wins.
        p_draw: Probability of a draw.
        p_away: Probability that team B wins.
        team_a: Name of first team.
        team_b: Name of second team.

    Returns:
        Tuple: (p_home, p_draw, p_away, modern_diff, modern_boost).
    """
    modern_diff = modern_strength(team_a) - modern_strength(team_b)
    modern_boost = 1 / (1 + np.exp(-modern_diff / 70))

    p_home *= modern_boost
    p_away *= 1 - modern_boost

    p_home, p_draw, p_away = _safe_normalize_probabilities(
        p_home,
        p_draw,
        p_away,
    )

    return p_home, p_draw, p_away, modern_diff, modern_boost

def _estimate_expected_goals_from_probabilities(
    p_home,
    p_draw,
    p_away,
    round_name='group',
):
    """Estimate expected goals from model probabilities.

    Args:
        p_home: Probability that team A wins.
        p_draw: Probability of a draw.
        p_away: Probability that team B wins.
        round_name: Tournament round name.

    Returns:
        Tuple: (expected_goals_a, expected_goals_b).
    """
    base_total_goals = GOALS_BY_ROUND.get(round_name, GOALS_BY_ROUND['group'])

    # High draw probability usually implies a tighter, lower-scoring match.
    total_expected_goals = base_total_goals * (1 - 0.35 * p_draw)
    total_expected_goals = np.clip(total_expected_goals, 1.50, 3.20)

    # Positive means team A is favoured; negative means team B is favoured.
    win_balance = p_home - p_away

    team_a_goal_share = 0.5 + 0.35 * win_balance
    team_a_goal_share = np.clip(team_a_goal_share, 0.25, 0.75)

    expected_goals_a = total_expected_goals * team_a_goal_share
    expected_goals_b = total_expected_goals * (1 - team_a_goal_share)

    return expected_goals_a, expected_goals_b

def _sample_result_from_probabilities(p_home, p_draw, p_away):
    """Sample one match outcome from model probabilities.

    Args:
        p_home: Probability that team A wins.
        p_draw: Probability of a draw.
        p_away: Probability that team B wins.

    Returns:
        One of 'A', 'D', or 'B'.
    """
    return np.random.choice(
        ['A', 'D', 'B'],
        p=[p_home, p_draw, p_away],
    )

def _sample_scoreline(expected_goals_a, expected_goals_b, result, max_goals=7):
    """Sample a scoreline consistent with the selected result.

    Args:
        expected_goals_a: Expected goals for team A.
        expected_goals_b: Expected goals for team B.
        result: One of 'A', 'B', or 'D'.
        max_goals: Maximum goals allowed per team.

    Returns:
        Tuple: (goals_a, goals_b).
    """
    for _ in range(500):
        goals_a = min(np.random.poisson(expected_goals_a), max_goals)
        goals_b = min(np.random.poisson(expected_goals_b), max_goals)

        if result == 'A' and goals_a > goals_b:
            return goals_a, goals_b

        if result == 'B' and goals_b > goals_a:
            return goals_a, goals_b

        if result == 'D' and goals_a == goals_b:
            return goals_a, goals_b

    if result == 'A':
        goals_b = min(np.random.poisson(expected_goals_b), max_goals - 1)
        goals_a = goals_b + 1
        return goals_a, goals_b

    if result == 'B':
        goals_a = min(np.random.poisson(expected_goals_a), max_goals - 1)
        goals_b = goals_a + 1
        return goals_a, goals_b

    goals = min(
        np.random.poisson((expected_goals_a + expected_goals_b) / 2),
        max_goals,
    )
    return goals, goals

def get_match_probabilities(team_a, team_b, model=model, round_name='group'):
    """Get match probabilities and expected goals for one fixture.

    Args:
        team_a: Name of first team.
        team_b: Name of second team.
        model: Trained model with predict_proba.
        round_name: Tournament round name.

    Returns:
        Dict with probabilities, expected goals, and feature diagnostics.
    """
    key = tuple(sorted([team_a, team_b]))
    h2h = h2h_stats.get(
        key,
        {
            'home_win_rate': 0.5,
            'draw_rate': 0.0,
            'matches_played': 1,
        },
    )

    fifa_diff_raw = fifa_points.get(team_a, 1500) - fifa_points.get(team_b, 1500)
    form_a = smooth_form(last5_win_rate.get(team_a, 0.5))
    form_b = smooth_form(last5_win_rate.get(team_b, 0.5))

    round_variance = ROUND_VARIANCE.get(round_name, ROUND_VARIANCE['group'])

    fifa_diff = fifa_diff_raw
    fifa_diff += np.random.normal(0, round_variance)
    fifa_diff += historical_penalty(team_a) - historical_penalty(team_b)

    h2h_weight = min(1.0, h2h['matches_played'] / 10)
    h2h_effective = h2h['home_win_rate'] * h2h_weight

    features = pd.DataFrame([{
        'fifa_diff': fifa_diff,
        'home_last5_win_rate': form_a,
        'away_last5_win_rate': form_b,
        'h2h_home_win_rate': h2h['home_win_rate'],
        'h2h_draw_rate': h2h['draw_rate'],
        'h2h_matches_played': h2h['matches_played'],
        'home_penalty_win_rate': penalty_win_rate.get(team_a, 0.5),
        'away_penalty_win_rate': penalty_win_rate.get(team_b, 0.5),
        'neutral': True,
        'fifa_diff_x_home_form': fifa_diff * form_a,
        'fifa_diff_x_away_form': fifa_diff * form_b,
        'h2h_effective': h2h_effective,
    }])

    p_home_raw, p_draw_raw, p_away_raw = _extract_model_probabilities(
        features,
        model=model,
    )

    p_home, p_draw, p_away, modern_diff, modern_boost = _apply_modern_adjustment(
        p_home=p_home_raw,
        p_draw=p_draw_raw,
        p_away=p_away_raw,
        team_a=team_a,
        team_b=team_b,
    )

    expected_goals_a, expected_goals_b = (
        _estimate_expected_goals_from_probabilities(
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            round_name=round_name,
        )
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
        'expected_goals': {
            team_a: expected_goals_a,
            team_b: expected_goals_b,
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
        },
    }

def simulate_match(
    team_a,
    team_b,
    model=model,
    round_name='group',
    return_only_result=False,
):
    """Simulate one match with random result and random scoreline.

    Args:
        team_a: Name of first team.
        team_b: Name of second team.
        model: Trained model with predict_proba.
        round_name: Tournament round name.
        return_only_result: If True, return only 'A', 'B', or 'D'.

    Returns:
        Match result string or full match dictionary.
    """
    match_info = get_match_probabilities(
        team_a=team_a,
        team_b=team_b,
        model=model,
        round_name=round_name,
    )

    p_home = match_info['probabilities']['team_a_win']
    p_draw = match_info['probabilities']['draw']
    p_away = match_info['probabilities']['team_b_win']

    result = _sample_result_from_probabilities(
        p_home=p_home,
        p_draw=p_draw,
        p_away=p_away,
    )

    expected_goals_a = match_info['expected_goals'][team_a]
    expected_goals_b = match_info['expected_goals'][team_b]

    goals_a, goals_b = _sample_scoreline(
        expected_goals_a=expected_goals_a,
        expected_goals_b=expected_goals_b,
        result=result,
    )

    if return_only_result:
        return result

    return {
        **match_info,
        'result': result,
        'winner': (
            team_a if result == 'A'
            else team_b if result == 'B'
            else None
        ),
        'goals_a': goals_a,
        'goals_b': goals_b,
        'score': f'{goals_a}-{goals_b}',
    }

def simulate_match_many(
    team_a,
    team_b,
    model=model,
    round_name='group',
    n_simulations=10000,
    use_tqdm=True,
):
    """Run many simulations of one match.

    Args:
        team_a: Name of first team.
        team_b: Name of second team.
        model: Trained model with predict_proba.
        round_name: Tournament round name.
        n_simulations: Number of simulations.

    Returns:
        DataFrame with one row per simulated match.
    """
    rows = []

    for _ in tqdm(range(n_simulations), desc=f'Simulating {team_a} vs {team_b}', total=n_simulations, unit='sim', disable=(not use_tqdm)):
        match = simulate_match(
            team_a=team_a,
            team_b=team_b,
            model=model,
            round_name=round_name,
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
            'team_a_win_probability': match['probabilities']['team_a_win'],
            'draw_probability': match['probabilities']['draw'],
            'team_b_win_probability': match['probabilities']['team_b_win'],
            'expected_goals_a': match['expected_goals'][team_a],
            'expected_goals_b': match['expected_goals'][team_b],
        })

    return pd.DataFrame(rows)

def get_result_from_goals(goals_a, goals_b):
    """Return match result from goals.

    Args:
        goals_a: Goals scored by team A.
        goals_b: Goals scored by team B.

    Returns:
        'A' if team A wins, 'B' if team B wins, otherwise 'D'.
    """
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
    """Score one tip against one simulated actual result.

    Args:
        tip_goals_a: Predicted goals for team A.
        tip_goals_b: Predicted goals for team B.
        actual_goals_a: Simulated actual goals for team A.
        actual_goals_b: Simulated actual goals for team B.
        phase: Tournament phase. Use 'group' for group-stage scoring.
            All later phases use knockout scoring:
            'play-in', 'round_of_16', 'quarterfinal', 'semifinal', 'final'.

    Returns:
        Integer points for this tip against this simulated result.
    """
    if phase == 'group':
        winner_points = 5
        home_goals_points = 1
        away_goals_points = 1
        goal_difference_points = 3
    elif phase in {
        'play-in',
        'round_of_16',
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
        tip_goals_a - tip_goals_b
        == actual_goals_a - actual_goals_b
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
    """Find the tip with the highest expected points from simulations.

    Args:
        df_simulations: DataFrame containing goals_a and goals_b columns.
        team_a: Name of team A.
        team_b: Name of team B.
        phase: Tournament phase. Use 'group' for group-stage scoring.
            All later phases use knockout scoring:
            'play-in', 'round_of_16', 'quarterfinal', 'semifinal', 'final'.
        max_tip_goals: Highest number of goals to consider in candidate tips.

    Returns:
        DataFrame with tips ranked by expected points.
    """
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