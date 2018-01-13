import luigi
import pandas as pd
import os
import datetime
import numpy as np

import jobs.historical.parse as parse
import jobs.model.optimize as optimize
import config as cfg


class RollingValues(luigi.Task):
    """
    Calculate the rolling mean and standard deviation at each time period
    """
    def requires(self):
        return [parse.FormatHistoricalBoxScores()]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PARSED, 'rolling-values.csv'))

    def run(self):
        with self.input()[0].open('r') as f:
            df = pd.read_csv(f)

        mean = df.set_index('GAME_ID')\
            .groupby('PLAYER')['PTS_DK']\
            .rolling(50, min_periods=1)\
            .mean()\
            .shift(1)\
            .reset_index()

        mean = pd.DataFrame(mean).set_index(['GAME_ID', 'PLAYER'])

        std = df.set_index('GAME_ID')\
            .groupby('PLAYER')['PTS_DK']\
            .rolling(50, min_periods=5)\
            .std()\
            .shift(1)\
            .reset_index()

        std = pd.DataFrame(std).set_index(['GAME_ID', 'PLAYER'])

        df = df.set_index(['GAME_ID', 'PLAYER'])

        df = df.join(mean, rsuffix='_MEAN')
        df = df.join(std, rsuffix='_STD')

        with self.output().open('w') as f:
            df.to_csv(f)


class SampleRotoFile(luigi.Task):
    """
    Create a sample augmented rotogrinders projection file for the date in question
    """
    test_date = pd.Timestamp(datetime.datetime(2017, 2, 1))

    def requires(self):
        return [RollingValues()]

    def output(self):
        return luigi.LocalTarget(os.path.join(
            cfg.DATA_PROJECTIONS, 'backtest', '{}_proj.csv'.format(self.test_date.strftime('%Y%m%d'))))

    def run(self):
        with self.input()[0].open('r') as f:
            df = pd.read_csv(f, parse_dates=['DATE'])

        proj = df[df['DATE'] == self.test_date]

        proj = proj.rename(columns={
            'PTS_DK_MEAN': 'PROJECTION',
        })[[
            'DATE',
            'PLAYER',
            'POSITION',
            'TEAM',
            'OPP',
            'VENUE',
            'PROJECTION',
            'PTS_DK_STD'
        ]]

        proj['PTS_DK_STD'].fillna(proj['PROJECTION'] / 2, inplace=True)

        proj['CEIL'] = proj['PROJECTION'] + proj['PTS_DK_STD']
        proj['FLOOR'] = proj['PROJECTION'] - proj['PTS_DK_STD']
        proj['SALARY'] = proj['PROJECTION'].apply(lambda x: max(round(x / np.random.normal(loc=0.0045, scale=0.00066), -2), 3000))

        with self.output().open('w') as f:
            proj.to_csv(f)


class SampleCorrelationFile(luigi.Task):
    """
    Create a sample correlations file
    """
    test_date = pd.Timestamp(datetime.datetime(2017, 2, 1))

    def requires(self):
        return [RollingValues()]

    def output(self):
        return luigi.LocalTarget(os.path.join(
            cfg.DATA_PROJECTIONS, 'backtest', '{}_dep.csv'.format(self.test_date.strftime('%Y%m%d'))))

    def run(self):
        with self.input()[0].open('r') as f:
            df = pd.read_csv(f, parse_dates=['DATE'])

        df = df[df['DATE'] <= self.test_date]
        players = df[df['DATE'] == self.test_date]['PLAYER'].unique()

        df = df[df['PLAYER'].isin(players)]
        df = df.set_index(['GAME_ID', 'TEAM'])
        df2 = df.copy()

        joined = df.join(df2, how='outer', rsuffix='_2').reset_index()
        joined = joined[joined['PLAYER'] < joined['PLAYER_2']]

        joined = joined[['TEAM', 'PLAYER', 'PLAYER_2', 'PTS_DK', 'PTS_DK_2']]

        min_games = 20
        group_by = ('TEAM', 'PLAYER', 'PLAYER_2')
        grp = joined.groupby(group_by).filter(lambda x: len(x) >= min_games).groupby(group_by)
        corr = grp[['PTS_DK', 'PTS_DK_2']].corr().iloc[0::2][['PTS_DK_2']]

        corr = corr.reset_index(group_by).reset_index()
        del corr['index']

        corr = corr.rename(columns={'PTS_DK_2': 'CORR'})

        with self.output().open('w') as f:
            corr.to_csv(f)


class EvaluateLineupSet(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())

    def requires(self):
        return [
            optimize.OptimizeLineupSet(date=self.date),
            parse.FormatHistoricalBoxScores(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PROJECTIONS,
                                              '{}_res.csv'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        entry = np.load(self.input()[0].path)

        with self.input()[1].open('r') as f:
            box_scores = pd.read_csv(f, parse_dates=['DATE'])

        comp_date = pd.Timestamp(self.date)

        box_scores = box_scores[box_scores['DATE'] == comp_date]
        box_scores = box_scores.sort_values(['TEAM', 'PLAYER']).reset_index()

        actual_points = box_scores['PTS_DK'].values
        players = box_scores['PLAYER'].values

        score_cols = ['P_{}_SCORE'.format(i) for i in range(entry.shape[1])]
        name_cols = ['P_{}_NAME'.format(i) for i in range(entry.shape[1])]

        lineup_results = pd.DataFrame(actual_points[entry], columns=score_cols)
        lineup_players = pd.DataFrame(players[entry], columns=name_cols)

        results = pd.concat((lineup_players, lineup_results), axis=1)

        results['TOTAL'] = results[score_cols].sum(axis=1)

        with self.output().open('w') as f:
            results.to_csv(f)
