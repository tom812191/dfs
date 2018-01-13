import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def iter_entries(dk_template, entry_names):
    """
    Generator function for lineup entries
    :param dk_template: pd.DataFrame with columns POSITION, NAME_AND_ID, NAME, ID
    :param entry_names: 2d numpy array of player names
    :yield: list of player ids in order for PG, SG, SF, PF, C, G, F, UTIL
    """
    for entry in entry_names:
        players = []
        for player in entry:

            row = dk_template[dk_template['NAME'] == player]

            try:
                assert len(row) == 1
            except AssertionError:
                if len(row) == 0:
                    raise AssertionError('Could not find player {} in dk_template'.format(player))
                raise AssertionError('Multiple entries for player {} in dk_template'.format(player))

            players.append({
                'position': row['POSITION'].values[0],
                'id': row['ID'].values[0],
            })

        yield order_players(players)


def order_players(players, positions=('PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'),
                  position_map=(('PG', 'G'), ('SG', 'G'), ('SF', 'F'), ('PF', 'F'), ('C', 'UTIL'))):
    """
    Put players in order of PG, SG, SF, PF, C, G, F, UTIL

    This is easily formulated as the Assignment Problem, so we can use the built in scipy package

    :param players: List of dictionaries of (position, id)
    :param positions: Positions in a lineup
    :param position_map: To be zipped into a dict, how primary positions map to secondary
    :return: ordered list of player ids
    """

    position_map = dict(position_map)

    # Create the cost matrix. Let players index the rows and positions index the columns
    cost_matrix = np.ones((len(players), len(positions)))
    util_idx = positions.index('UTIL')

    assert len(players) == len(positions)

    # Construct the cost matrix. If a player can go in the position, cost is 0, otherwise keep the cost at 1
    for player_idx, player in enumerate(players):
        player_positions = player['position'].split('/')
        for position in player_positions:
            position_idx = positions.index(position)
            secondary_position_idx = positions.index(position_map[position])

            cost_matrix[player_idx, position_idx] = 0
            cost_matrix[player_idx, secondary_position_idx] = 0
            cost_matrix[player_idx, util_idx] = 0

    # Solve the assignment problem
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    # Reorder the players accordingly
    results = [None] * len(positions)
    for player_idx, position_idx in enumerate(col_idx):
        results[position_idx] = players[player_idx]['id']

    assert None not in results
    return results


def calculate_exposure(dk_template, entry_names):
    """
    Calculate the exposures of all players in dk_template
    :param dk_template: pd.DataFrame with columns POSITION, NAME_AND_ID, NAME, ID
    :param entry_names: 2d numpy array of player names
    :return: dk_templates with added exposure column
    """
    dk_template = dk_template[['NAME', 'POSITION']]
    dk_template = dk_template.set_index('NAME', drop=False)

    exposure = pd.Series(entry_names.flatten()).value_counts() / len(entry_names)
    exposure = exposure.to_frame('EXPOSURE')

    out = dk_template.join(exposure).fillna(0.0).sort_values('EXPOSURE', ascending=False)

    return out
