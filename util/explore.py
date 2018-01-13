import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

import pylab


def get_distributions(df, min_minutes=12, min_games=50, dfs_pts_col='PTS_DK'):

    # Filter to games where the player played at least min_minutes
    df = df[df['MIN'] > min_minutes]

    # Roll dfs scores up to a list for each player
    pts = df.groupby('PLAYER')[dfs_pts_col].apply(np.array)
    pts = pd.DataFrame(pts)

    # Filter on min games
    pts['num_games'] = pts[dfs_pts_col].apply(lambda x: len(x))
    pts = pts[pts['num_games'] > min_games]

    # Fit Distribution
    pts['dist'] = pts[dfs_pts_col].apply(fit_distribution)
    pts['best_dist'] = pts['dist'].apply(lambda x: x[0]['dist'])
    pts['best_dist_pvalue'] = pts['dist'].apply(lambda x: x[0]['kstest_pvalue'])

    # Get descriptive stats
    pts['mean'] = pts[dfs_pts_col].apply(np.mean)
    pts['std'] = pts[dfs_pts_col].apply(np.std)
    pts['skew'] = pts[dfs_pts_col].apply(stats.skew)
    pts['kurtosis'] = pts[dfs_pts_col].apply(stats.kurtosis)

    del pts[dfs_pts_col]

    return pts


def fit_distribution(x, choices=(
        'gamma',
        'lognorm',)
                     ):

    results = []
    for dist_name in choices:
        dist = getattr(stats, dist_name)
        param = dist.fit(x)
        results.append({
            'params': param,
            'dist': dist_name,
            'kstest_pvalue': stats.kstest(x, dist_name, args=param)[1]
        })

    return sorted(results, key=lambda k: k['kstest_pvalue'], reverse=True)


def qqplot(df, player, dist_name, dfs_pts_col='PTS_DK'):
    pts = df.loc[player, dfs_pts_col]
    dist = getattr(stats, dist_name)
    stats.probplot(pts, sparams=dist.fit(pts), dist=dist_name, plot=pylab)


def dependence(df, player_1_name, player_2_name, dfs_pts_col='PTS_DK'):
    df = df.set_index('GAME_ID')
    p1 = df[df['PLAYER'] == player_1_name]
    p2 = df[df['PLAYER'] == player_2_name]

    rsuffix = '_2'
    joined = p1.join(p2, rsuffix=rsuffix, how='inner')
    joined = joined[joined['MIN'] > 12]
    joined = joined[joined['MIN{}'.format(rsuffix)] > 12]

    vals = joined[[dfs_pts_col, '{}{}'.format(dfs_pts_col, rsuffix)]].values

    return vals


def plot_dependence(df, player_1_name, player_2_name, dfs_pts_col='PTS_DK'):
    vals = dependence(df, player_1_name, player_2_name, dfs_pts_col=dfs_pts_col)
    # plt.scatter(vals[:, 0], vals[:, 1])

    ranks = stats.mstats.rankdata(vals, axis=0)
    plt.scatter(ranks[:, 0], ranks[:, 1])

    print(np.corrcoef(vals[:, 0], vals[:, 1]))


def average_scores(df):
    df = df[df['MIN'] > 12]
    return df.groupby('PLAYER').agg({'PTS_DK': 'mean'})


def pairwise_scores(df):
    df = df.set_index('GAME_ID')
    df2 = df.copy()

    joined = df.join(df2, how='outer', rsuffix='_2')
    joined = joined[joined['PLAYER'] < joined['PLAYER_2']]

    joined = joined.rename(columns={
        'PLAYER': 'PLAYER_1',
        'POSITION': 'POSITION_1',
        'TEAM': 'TEAM_1',
        'PTS_DK': 'PTS_DK_1',
        'PTS_DK_AVG': 'PTS_DK_AVG_1',
        'TEAM_RANK': 'TEAM_RANK_1',
        'POSITION_RANK': 'POSITION_RANK_1',
    })

    joined['SAME_TEAM'] = joined['TEAM_1'] == joined['TEAM_2']

    return joined[[
        'PLAYER_1',
        'PLAYER_2',
        'TEAM_1',
        'TEAM_2',
        'POSITION_1',
        'POSITION_2',
        'PTS_DK_1',
        'PTS_DK_2',
        'PTS_DK_AVG_1',
        'PTS_DK_AVG_2',
        'TEAM_RANK_1',
        'TEAM_RANK_2',
        'POSITION_RANK_1',
        'POSITION_RANK_2',
        'SAME_TEAM',
    ]]


def pairwise(df):
    df = df.set_index('GAME_ID')
    df2 = df.copy()

    joined = df.join(df2, how='outer', rsuffix='_2')
    joined = joined[joined['PLAYER'] < joined['PLAYER_2']]

    joined = joined.rename(columns={
        'PLAYER': 'PLAYER_1',
        'POSITION': 'POSITION_1',
        'TEAM': 'TEAM_1',
        'PTS_DK': 'PTS_DK_1',
    })

    joined['SAME_TEAM'] = joined['TEAM_1'] == joined['TEAM_2']

    return joined[[
        'PLAYER_1',
        'PLAYER_2',
        'TEAM_1',
        'TEAM_2',
        'POSITION_1',
        'POSITION_2',
        'PTS_DK_1',
        'PTS_DK_2',
        'SAME_TEAM',
    ]]


def correlations(df, min_games=50, group_by=('PLAYER_1', 'PLAYER_2')):
    df = df.groupby(group_by).filter(lambda x: len(x) >= min_games)
    grp = df.groupby(group_by)

    corr = grp[['PTS_DK_1', 'PTS_DK_2']].corr().iloc[0::2][['PTS_DK_2']]

    corr = corr.reset_index(group_by).reset_index()
    del corr['index']

    corr = corr.rename(columns={'PTS_DK_2': 'PTS_DK'})

    return corr


def process_pairwise(df, group_by=('PLAYER_1', 'PLAYER_2', 'TEAM_1', 'TEAM_2'), min_games=50):

    class Data:
        def __init__(self, data):
            self.data = data

    def unique_list(series):
        return ','.join([str(d) for d in series.unique().tolist()])

    def corr(series):
        vals = np.array(series.values.tolist())
        return np.corrcoef(vals[:, 0], vals[:, 1])[0, 1]

    def ranks(series):
        vals = np.array(series.values.tolist())
        return Data([stats.rankdata(vals[:, 0]).tolist(), stats.rankdata(vals[:, 1]).tolist()])

    df = df.copy()

    df['CORR'] = df.apply(lambda row: [row['PTS_DK_1'], row['PTS_DK_2']], axis=1)
    df['RANKS'] = df['CORR']

    df = df.groupby(group_by).filter(lambda x: len(x) >= min_games)

    df = df.groupby(group_by).agg({
        'PLAYER_1': unique_list,
        'PLAYER_2': unique_list,
        'TEAM_1': unique_list,
        'TEAM_2': unique_list,
        'POSITION_1': unique_list,
        'POSITION_2': unique_list,
        'TEAM_RANK_1': unique_list,
        'TEAM_RANK_2': unique_list,
        'POSITION_RANK_1': unique_list,
        'POSITION_RANK_2': unique_list,
        'SAME_TEAM': unique_list,
        'CORR': corr,
        'RANKS': ranks,

    })

    return df.reset_index(drop=True).sort_values('CORR')


def plot_scatter(df, idx):
    a = np.array(df.iloc[idx].RANKS.data).transpose()
    plt.scatter(a[:, 0], a[:, 1])


def reorder_columns_by_position(row):
    """
    Call from df.apply(reorder_columns_by_position, axis=1)
    :param row:
    :return:
    """
    if row.POSITION_1 <= row.POSITION_2:
        return row

    cols = ['POSITION', 'PLAYER', 'TEAM', 'PTS_DK', 'PTS_DK_AVG', 'TEAM_RANK', 'POSITION_RANK']
    for col in cols:
        col_1 = '{}{}'.format(col, '_1')
        col_2 = '{}{}'.format(col, '_2')

        tmp = row[col_1]
        row[col_1] = row[col_2]
        row[col_2] = tmp

    return row


def reorder_columns_by_team_rank(row):
    """
    Call from df.apply(reorder_columns_by_position, axis=1)
    :param row:
    :return:
    """
    if row.TEAM_RANK_1 <= row.TEAM_RANK_2:
        return row

    cols = ['POSITION', 'PLAYER', 'TEAM', 'PTS_DK', 'PTS_DK_AVG', 'TEAM_RANK', 'POSITION_RANK']
    for col in cols:
        col_1 = '{}{}'.format(col, '_1')
        col_2 = '{}{}'.format(col, '_2')

        tmp = row[col_1]
        row[col_1] = row[col_2]
        row[col_2] = tmp

    return row
