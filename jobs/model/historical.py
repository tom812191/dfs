import os
import luigi
import pandas as pd
import glob

import config as cfg

import jobs.historical.parse as parse


class ParseHistoricalScores(luigi.Task):
    """
    Calculate the correlation between two players at the given time period
    """

    def requires(self):
        return [parse.FormatHistoricalBoxScores()]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PARSED, 'box-scores.csv'))

    def run(self):
        with self.input()[0].open('r') as f:
            df = pd.read_csv(f)