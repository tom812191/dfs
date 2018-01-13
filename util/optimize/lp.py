import numpy as np
import pulp as lp


def generate_lineup_universe(proj, num_lineups=1000, sample_frac=0.8, progress_cb=None):
    df = prep_data_for_lp(proj)

    lineups = [list(optimize_lineup_lp(df).index)]
    exposure = np.ones(len(df))
    for i in range(num_lineups):
        if progress_cb is not None:
            progress_cb(i/num_lineups)

        weights = 1 / exposure
        sample = df.sample(frac=sample_frac, weights=weights)
        objective_field = 'PROJECTION' if i % 2 else 'CEIL'
        lineup = optimize_lineup_lp(sample, objective_field=objective_field)
        indexes = list(lineup.index)
        lineups.append(indexes)
        exposure[indexes] += 1

    lineups = np.array(lineups)
    return np.unique(np.sort(lineups), axis=0)


def prep_data_for_lp(df):
    df = df.copy().sort_values(['TEAM', 'PLAYER']).reset_index()

    # Create binary columns for each position to represent if player has the position
    df['PG'] = df['POSITION'].apply(lambda p: int('1' in str(p)))
    df['SG'] = df['POSITION'].apply(lambda p: int('2' in str(p)))
    df['SF'] = df['POSITION'].apply(lambda p: int('3' in str(p)))
    df['PF'] = df['POSITION'].apply(lambda p: int('4' in str(p)))
    df['C'] = df['POSITION'].apply(lambda p: int('5' in str(p)))

    return df


def optimize_lineup_lp(df, max_salary=50000, objective_field='PROJECTION'):
    """
    Maximize the mean of a single lineup subject to draftkings lineup constraints
    :param df: pd.DataFrame with projections that's already been prepped
    :param max_salary: The maximum allowed salary
    :param objective_field: The field in df that should be maximized. Should be PROJECTION, CEIL, or FLOOR
    """

    # Create binary variables and model
    x = lp.LpVariable.dicts('x', df.index, lowBound=0, upBound=1, cat='Integer')
    mod = lp.LpProblem('Lineup', lp.LpMaximize)

    # Objective function: Maximize Proj Pts
    objvals = {idx: df[objective_field][idx] for idx in df.index}
    mod += sum([x[idx] * objvals[idx] for idx in df.index])

    # Salary constraint: Total salary < 50,000
    mod += sum([x[idx] * df['SALARY'][idx] for idx in df.index]) <= max_salary

    """
    Players constrains

    Note that a single player can qualify for multiple positions
    """

    # Total player constraint: Exactly 8 total players
    mod += sum([x[idx] for idx in df.index]) == 8

    # At least 1 of each position
    mod += sum([x[idx] * df['PG'][idx] for idx in df.index]) >= 1
    mod += sum([x[idx] * df['SG'][idx] for idx in df.index]) >= 1
    mod += sum([x[idx] * df['SF'][idx] for idx in df.index]) >= 1
    mod += sum([x[idx] * df['PF'][idx] for idx in df.index]) >= 1
    mod += sum([x[idx] * df['C'][idx] for idx in df.index]) >= 1

    # At least 3 G and 3 F total
    mod += sum([x[idx] * (df['PG'][idx] or df['SG'][idx]) for idx in df.index]) >= 3
    mod += sum([x[idx] * (df['SF'][idx] or df['PF'][idx]) for idx in df.index]) >= 3
    mod += sum([x[idx] * (df['PF'][idx] or df['C'][idx]) for idx in df.index]) >= 2

    # Additional for multiple eligibility
    mod += sum([x[idx] * (df['PF'][idx] or df['SF'][idx] or df['C'][idx]) for idx in df.index]) >= 4
    mod += sum([x[idx] * (df['PF'][idx] and not df['SG'][idx]) for idx in df.index]) <= 3

    # Solve lp
    mod.solve()

    # Output solution
    solution = []
    for idx in df.index:
        if x[idx].value() > 0:
            solution.append(idx)

    lineup = df.loc[solution, ['PLAYER', 'POSITION', 'PROJECTION', 'SALARY']]

    return lineup
