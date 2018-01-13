import numpy as np


def evaluate_lineup_set(lineup_universe, simulated_scores, lineup_set_indexes,
                        target=330, duplicate_penalty_factor=0.5, as_tuple=True):
    """
    Calculate the fitness of the lineup set as the average number of simulations where at least 1 lineup hits the target

    Add a penalty if any of the lineups are the same

    :param lineup_universe: All lineups
    :param simulated_scores: Simulated scores for all players, the first dimension is players, the second is sim number
    :param lineup_set_indexes: The indexes of the lineups from lineup_universe
    :param target: Target score to win the contest
    :param duplicate_penalty_factor: A factor applied to the final score to penalize sets with duplicate entries, such
    that for penalty p, score = score * p

    :return: A single fitness score in 0..1
    """

    penalty_factor = 1
    if len(lineup_set_indexes) > len(np.unique(lineup_set_indexes)):
        penalty_factor = duplicate_penalty_factor

    lineups = lineup_universe[lineup_set_indexes]
    scores = simulated_scores[lineups]

    hit_target = scores.sum(axis=1) > target
    pct = float(hit_target.any(axis=0).mean())

    output = pct * penalty_factor

    if as_tuple:
        return output,

    return output

