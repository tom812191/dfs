import numpy as np
from scipy import linalg, stats


def simulate_slate(proj, dep, n):
    proj = proj.sort_values(['TEAM', 'PLAYER'])
    teams = proj['TEAM'].unique().tolist()

    player_scores = None
    for team in teams:
        print(team)
        scores = simulate_team(proj, dep, team, n)

        if player_scores is None:
            player_scores = scores
        else:
            player_scores = np.concatenate((player_scores, scores), axis=0)

    return player_scores


def simulate_team(proj, dep, team, n):
    proj = proj[proj['TEAM'] == team].sort_values('PLAYER')

    players = proj['PLAYER'].unique().tolist()
    d = len(players)

    dep = dep[dep['PLAYER_1'].isin(players) & dep['PLAYER_2'].isin(players)]

    corr = np.identity(d)
    for row in dep.iterrows():
        row = row[1]
        i = players.index(row['PLAYER_1'])
        j = players.index(row['PLAYER_2'])
        c = row['PTS_DK']
        corr[i, j] = c
        corr[j, i] = c

    # A is the lower triangular matrix of Cholesky decomp of the correlation matrix
    try:
        A = linalg.cholesky(corr, lower=True)
    except np.linalg.linalg.LinAlgError as e:
        # The matrix is not positive definite, so convert to the nearest positive definite matrix
        A = linalg.cholesky(nearest_positive_semi_definite(corr), lower=True)

    # Z is a dxn matrix of random standard normal variables
    Z = np.random.normal(size=(d, n))

    # X is a dxn matrix of the correlated standard random normal variables
    X = np.dot(A, Z)

    # U is a dxn matrix of uniform values to be plugged into the marginal distributions
    U = stats.norm.cdf(X)

    # scale = var / mean
    # a = mean**2 / var
    player_mean = proj['PROJECTION'].values
    player_var = (proj['CEIL'].values - player_mean) ** 2

    scale = player_var / player_mean
    a = player_mean ** 2 / player_var

    if d != len(scale):
        print(d)
        print(len(scale))
        print(team)

    assert d == len(scale)

    a = a.reshape((d, 1))
    scale = scale.reshape((d, 1))

    return stats.gamma.ppf(U, a, scale=scale)


def nearest_positive_semi_definite(A, epsilon=0.00001):
    n = A.shape[0]
    eigval, eigvec = np.linalg.eig(A)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1 / (np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)))))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    out = B * B.T
    return out
