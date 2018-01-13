import pandas as pd


def dfs_points(scoring, row):
    double_count = (row.TOT_R >= 10) + \
                   (row.A >= 10) + \
                   (row.ST >= 10) + \
                   (row.BL >= 10) + \
                   (row.PTS >= 10)

    total = 0
    for key in scoring:
        if key == 'DD':
            total += scoring[key] * (double_count >= 2)
        elif key == 'TD':
            total += scoring[key] * (double_count >= 3)
        else:
            total += row[key] * scoring[key]

    return total


def game_id(row):
    home = row.OPP if row.VENUE == 'R' else row.TEAM
    away = row.OPP if row.VENUE == 'H' else row.TEAM
    date = row.DATE.strftime('%Y%m%d')

    return '{}{}{}'.format(date, away, home)


def scrub_position(mapping, row):
    position = row.POSITION

    if pd.isnull(position):
        position = mapping['PLAYER_TO_POSITION'][row.PLAYER]

    return mapping['POSITION_TO_POSITION'][position]


def map_position(mapping, pos):
    positions = [str(mapping[p]) for p in pos.split('/')]
    return ','.join(positions)


def ranks(df):
    """
    Calculate the players in-game ranking for his team and his team for his position

    E.g. If Team A has players A_PG, A_SF, A_C with fantasy points 1, 3, 5 respectively, and Team B has players
    B_PG, B_SF, B_C with fantasy points 2, 4, 6 respectively, then we'll have

    GAME_ID | PLAYER | TEAM RANK | POSITION RANK
          1 |   A_PG |         3 |             1
          1 |   A_SF |         2 |             1
          1 |    A_C |         1 |             1
          1 |   B_PG |         3 |             1
          1 |   B_SF |         2 |             1
          1 |    B_C |         1 |             1


    :param df: The Box Scores data set

    :return: DataFrame with added columns
    """
    qualifying_games = df[df['MIN'] > 12]
    player_averages = qualifying_games.groupby('PLAYER').agg({'PTS_DK': 'mean'})

    df = df.set_index('PLAYER').join(player_averages, rsuffix='_AVG')

    df['PTS_DK_AVG'] = df['PTS_DK_AVG'].fillna(0)

    df = df.reset_index()

    df['TEAM_RANK'] = df.groupby(('GAME_ID', 'TEAM'))['PTS_DK_AVG'].rank(ascending=False)
    df['POSITION_RANK'] = df.groupby(('GAME_ID', 'TEAM', 'POSITION'))['PTS_DK_AVG'].rank(ascending=False)

    return df



