import os

DATA_ROOT = '/Users/tom/Projects/Portfolio/data/dfs'
# /Users/tom/Projects/Portfolio/data/dfs/parsed/pairwise.csv
# /Users/tom/Projects/Portfolio/data/dfs/parsed/box-scores-formatted.csv
# /Users/tom/Projects/Portfolio/data/dfs/viz/dependence_by_position_rank.csv

DATA_RAW = os.path.join(DATA_ROOT, 'raw')
DATA_PARSED = os.path.join(DATA_ROOT, 'parsed')
DATA_VIZ = os.path.join(DATA_ROOT, 'viz')
DATA_PROJECTIONS = os.path.join(DATA_ROOT, 'projections')
DATA_TMP = os.path.join(DATA_ROOT, 'tmp')

DFS_SCORING = {
    'DK': {
        'PTS': 1,
        '3P': 0.5,
        'TOT_R': 1.25,
        'A': 1.5,
        'ST': 2,
        'BL': 2,
        'TO': -0.5,
        'DD': 1.5,
        'TD': 3,
    }
}

DFS_LINEUP = {
    'DK': {
        'PG': [1],
        'SG': [2],
        'SF': [3],
        'PF': [4],
        'C': [5],
        'G': [1, 2],
        'F': [3, 4],
        'Util': [1, 2, 3, 4, 5],
    }
}

DFS_CONSTRAINTS = {
    'DK': {
        'SALARY': 50000,
        'MIN_GAMES': 2,
    }
}

MARGINAL_DIST = 'gamma'

POSITION_MAP = {
    'POSITION_TO_POSITION': {
        'PG': 1,
        'SG': 2,
        'SF': 3,
        'PF': 4,
        'C': 5,
        'F': 3,
        'SF-PF': 3,
        'G': 2,
        'F-C': 4,
        'G-F': 3,
    },

    'PLAYER_TO_POSITION': {
        'Roger Mason Jr.': 'SF',
        'DJ White': 'PF',
        'Michael Harris': 'SF',
        'Hamady Ndiaye': 'C',
        'DJ Stephens': 'SG',
    }
}
