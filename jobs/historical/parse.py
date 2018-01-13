import os
import luigi
import pandas as pd
import glob

import config as cfg

import util.parse as parse
import functools


class ParseHistoricalBoxScores(luigi.Task):
    """
    Parse the historical box score xlsx files into a single csv
    """

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PARSED, 'box-scores.csv'))

    def run(self):
        box_scores = []

        # Read in box score and team sheets from each XLSX file
        for path in glob.glob(os.path.join(cfg.DATA_RAW, '*.xlsx')):
            with open(path, 'rb') as f:
                box_scores.append(pd.read_excel(f))

            if '2017' in path:
                with open(path, 'rb') as f:
                    teams = pd.read_excel(f, sheet_name='Teams-Cities').rename(columns={
                        'NBA.com Abbreviation': 'TEAM_CODE',
                        'City Name': 'CITY'
                    })

        # Concat data frames
        box_score = pd.concat(box_scores, ignore_index=True)
        del box_scores

        # Map team names
        teams = teams[['TEAM_CODE', 'CITY']].drop_duplicates(subset=['CITY'], keep='last').set_index('CITY')

        box_score = box_score\
            .join(teams, on='OWN TEAM')\
            .join(teams, on='OPP TEAM', rsuffix='_OPP')\
            .rename(columns={
                'TEAM_CODE': 'TEAM',
                'TEAM_CODE_OPP': 'OPP',
                'VENUE (R/H)': 'VENUE',
                'PLAYER FULL NAME': 'PLAYER',
                'TOT': 'TOT_R',
            })

        # Select Needed Columns
        box_score = box_score[[
            'DATE',
            'PLAYER',
            'POSITION',
            'TEAM',
            'OPP',
            'VENUE',
            'MIN',
            'FG',
            'FGA',
            '3P',
            '3PA',
            'FT',
            'FTA',
            'OR',
            'DR',
            'TOT_R',
            'A',
            'PF',
            'ST',
            'TO',
            'BL',
            'PTS',
        ]]

        # write to csv
        with self.output().open('w') as f:
            box_score.to_csv(f, index=False)


class FormatHistoricalBoxScores(luigi.Task):
    """
    Format box scores with draftkings and fanduel scores and a consistent game id
    """

    def requires(self):
        return [
            ParseHistoricalBoxScores()
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PARSED, 'box-scores-formatted.csv'))

    def run(self):
        with self.input()[0].open('r') as f:
            df = pd.read_csv(f, parse_dates=['DATE'])

        pts_dk = functools.partial(parse.dfs_points, cfg.DFS_SCORING['DK'])

        # Calculate GAME_ID and PTS_DK
        df['GAME_ID'] = df.apply(parse.game_id, axis=1)
        df['PTS_DK'] = df.apply(pts_dk, axis=1)

        # Select Columns
        df = df[[
            'GAME_ID',
            'DATE',
            'PLAYER',
            'POSITION',
            'TEAM',
            'OPP',
            'VENUE',
            'MIN',
            'PTS_DK',
        ]]

        # Scrub Positions
        scrub_position = functools.partial(parse.scrub_position, cfg.POSITION_MAP)
        df['POSITION'] = df.apply(scrub_position, axis=1)

        # Calculate Ranks
        # df = parse.ranks(df)

        # Write to csv
        with self.output().open('w') as f:
            df.to_csv(f, index=False)


class RankBoxScores(luigi.Task):
    """
    Format box scores with draftkings and fanduel scores and a consistent game id
    """

    def requires(self):
        return [
            FormatHistoricalBoxScores()
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PARSED, 'box-scores-ranked.csv'))

    def run(self):
        with self.input()[0].open('r') as f:
            df = pd.read_csv(f)

        # Calculate Ranks
        df = parse.ranks(df)

        # Write to csv
        with self.output().open('w') as f:
            df.to_csv(f, index=False)
