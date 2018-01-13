import os
import luigi
import pandas as pd
import glob

import config as cfg

import util.explore as explore
from jobs.historical.parse import RankBoxScores


class ExploreDistributions(luigi.Task):
    def requires(self):
        return [RankBoxScores()]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PARSED, 'distributions.csv'))

    def run(self):
        with self.input()[0].open('r') as f:
            df = pd.read_csv(f)

        df = explore.get_distributions(df)

        with self.output().open('w') as f:
            df.to_csv(f, index=True)


class PairwiseResults(luigi.Task):
    def requires(self):
        return [RankBoxScores()]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PARSED, 'pairwise.csv'))

    def run(self):
        with self.input()[0].open('r') as f:
            df = pd.read_csv(f)

        df = explore.pairwise_scores(df)

        df = df.reset_index()

        with self.output().open('w') as f:
            df.to_csv(f)


