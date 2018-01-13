import luigi
import os
import pandas as pd
import pickle

from jobs.model import data
from util import target
import config


class ScoreTargetLinearModel(luigi.Task):
    def requires(self):
        return [data.ScoreTargetData()]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_ROOT, 'static', 'ScoreTargetModel.p'))

    def run(self):
        with self.input()[0].open('r') as f:
            df = pd.read_csv(f)

        X = df[['Contest Size', 'Number of Games']].values
        y = df[['Winning Score']].values
        model = target.linear_model(X, y)

        with open(self.output().path, 'wb') as f:
            pickle.dump(model, f)
