import luigi
import datetime
import os
import pandas as pd
from functools import partial
from requests import get

import config
from jobs.historical import parse
from util import parse as parseutil
from util import explore


class ProjectionData(luigi.task.ExternalTask):
    date = luigi.DateParameter(default=datetime.date.today())

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_PROJECTIONS,
                                              '{}_proj.csv'.format(self.date.strftime('%Y%m%d'))))


class DependencyData(luigi.task.ExternalTask):
    date = luigi.DateParameter(default=datetime.date.today())

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_PROJECTIONS,
                                              '{}_dep.csv'.format(self.date.strftime('%Y%m%d'))))


class DKUploadTemplate(luigi.task.ExternalTask):
    date = luigi.DateParameter(default=datetime.date.today())

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_RAW,
                                              '{}_DKSalaries.csv'.format(self.date.strftime('%Y%m%d'))))


class CityTeamMap(luigi.task.ExternalTask):
    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_RAW, 'city-team-map.csv'))


class ParseDependencyData(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())

    def requires(self):
        return [
            InSeasonData(),
            parse.FormatHistoricalBoxScores(),
            CityTeamMap(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_PROJECTIONS,
                                              '{}_dep.csv'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        with self.input()[0].open('r') as f:
            in_season = pd.read_csv(f, parse_dates=['DATE'])

        with self.input()[1].open('r') as f:
            box_scores = pd.read_csv(f, parse_dates=['DATE'])

        with self.input()[2].open('r') as f:
            city_team_map = pd.read_csv(f, index_col='CITY')

        map_position = partial(parseutil.map_position, config.POSITION_MAP['POSITION_TO_POSITION'])

        in_season.loc[in_season['POSITION_DK'].isna(), 'POSITION_DK'] = \
            in_season[in_season['POSITION_DK'].isna()]['POSITION_FD']

        in_season.loc[in_season['POSITION_DK'].isna(), 'POSITION_DK'] = \
            in_season[in_season['POSITION_DK'].isna()]['POSITION_YA']

        in_season = in_season[~in_season['POSITION_DK'].isna()]

        in_season['POSITION'] = in_season['POSITION_DK'].apply(map_position)

        in_season['TEAM'] = in_season['TEAM_CITY'].apply(lambda city: city_team_map.loc[city, 'TEAM'])
        in_season['OPP'] = in_season['OPP_CITY'].apply(lambda city: city_team_map.loc[city, 'TEAM'])
        in_season['GAME_ID'] = in_season.apply(parseutil.game_id, axis=1)

        box_scores = box_scores[['GAME_ID', 'PLAYER', 'POSITION', 'TEAM', 'PTS_DK']]
        in_season = in_season[['GAME_ID', 'PLAYER', 'POSITION', 'TEAM', 'PTS_DK']]

        scores = pd.concat([box_scores, in_season], ignore_index=True, axis=0)
        pairwise = explore.pairwise(scores)

        correlations = explore.correlations(pairwise, group_by=('PLAYER_1', 'PLAYER_2', 'SAME_TEAM',), min_games=20)
        correlations = correlations[correlations['SAME_TEAM']]

        with self.output().open('w') as f:
            correlations.to_csv(f)


class InSeasonDataRaw(luigi.task.ExternalTask):

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_RAW, 'season-dfs-stats.xlsx'))


class InSeasonData(luigi.task.Task):

    def requires(self):
        return [InSeasonDataRaw()]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_RAW, 'season_dfs_stats.csv'))

    def run(self):
        columns = [
            'DATASET',
            'DATE',
            'PLAYER',
            'TEAM_CITY',
            'OPP_CITY',
            'VENUE',
            'MINUTES',
            'USAGE',
            'POSITION_DK',
            'POSITION_FD',
            'POSITION_YA',
            'SALARY_DK',
            'SALARY_FD',
            'SALARY_YA',
            'PTS_DK',
            'PTS_FD',
            'PTS_YA',
        ]

        df = pd.read_excel(self.input()[0].path, sheet_name='DFS Stats', skiprows=2, header=None, names=columns)

        with self.output().open('w') as f:
            df.to_csv(f)


class DownloadProjectionData(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())

    def requires(self):
        return [DKUploadTemplate(date=self.date)]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_PROJECTIONS,
                                              '{}_proj.csv'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        tmp_file = os.path.join(config.DATA_TMP, 'tmp-rotogrinders-projections.csv')
        url = 'https://rotogrinders.com/projected-stats/nba-player.csv?site=draftkings'
        with open(tmp_file, 'wb') as f:
            response = get(url)
            f.write(response.content)

        with open(tmp_file, 'r') as f:
            df = pd.read_csv(f, header=None,
                             names=['PLAYER_RG', 'SALARY_RG', 'TEAM', 'POSITION', 'OPP', 'CEIL', 'FLOOR', 'PROJECTION'])

        df['PTS_DK_STD'] = df['CEIL'] - df['PROJECTION']

        map_position = partial(parseutil.map_position, config.POSITION_MAP['POSITION_TO_POSITION'])

        df['POSITION'] = df['POSITION'].apply(map_position)

        # Read in the DK Template and use these salaries, and filter to only players in the template
        with self.input()[0].open('r') as f:
            dk_template = pd.read_csv(f, skiprows=7)

        dk_template = dk_template.iloc[:, 8:14].reset_index(drop=True)
        dk_template.columns = ['POSITION', 'NAME_AND_ID', 'PLAYER', 'ID', 'xxx', 'SALARY']
        dk_template = dk_template[['PLAYER', 'SALARY']]

        df = df.set_index('PLAYER_RG')
        projections = dk_template.join(df, on='PLAYER', rsuffix='_RG')

        projections = projections.dropna()

        with self.output().open('w') as f:
            projections.to_csv(f)

        # Delete tmp file
        os.remove(tmp_file)


class ScoreTargetData(luigi.task.ExternalTask):
    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_RAW, 'ScoreTarget.csv'))
