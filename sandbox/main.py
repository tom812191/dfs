import pandas as pd
import os
import config

from util import explore


def main():
    pw = pd.read_csv(os.path.join(config.DATA_PARSED, 'pairwise.csv'))
    pw = pw.apply(explore.reorder_columns_by_team_rank, axis=1)

    pw = pw[pw['TEAM_RANK_1'] <= 7]
    pw = pw[pw['TEAM_RANK_2'] <= 7]

    pw['PTS_RANK_1'] = pw.groupby(('TEAM_RANK_1', 'TEAM_RANK_2'))['PTS_DK_1'].rank(ascending=False, pct=True)
    pw['PTS_RANK_2'] = pw.groupby(('TEAM_RANK_1', 'TEAM_RANK_2'))['PTS_DK_2'].rank(ascending=False, pct=True)

    pw.to_csv(os.path.join(config.DATA_VIZ, 'dependence_by_team_rank.csv'))

    


if __name__ == '__main__':
    main()
