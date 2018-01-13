import luigi
import datetime
import os
import pandas as pd
import numpy as np

from jobs.model import data
from util import simulate
import config


class SimulateSlate(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())
    num_simulations = luigi.IntParameter(default=10000)

    def requires(self):
        return [
            data.DownloadProjectionData(date=self.date),
            data.ParseDependencyData(date=self.date),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_PROJECTIONS,
                                              '{}_scores.npy'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        with self.input()[0].open('r') as f:
            proj = pd.read_csv(f)

        with self.input()[1].open('r') as f:
            dep = pd.read_csv(f)

        simulated_scores = simulate.simulate_slate(proj, dep, self.num_simulations)

        np.save(self.output().path, simulated_scores)


class SimulateCrossValidationSlate(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())
    num_simulations = luigi.IntParameter(default=1000000)

    def requires(self):
        return [
            data.DownloadProjectionData(date=self.date),
            data.ParseDependencyData(date=self.date),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_PROJECTIONS,
                                              '{}_scores_cross_validation.npy'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        with self.input()[0].open('r') as f:
            proj = pd.read_csv(f)

        with self.input()[1].open('r') as f:
            dep = pd.read_csv(f)

        simulated_scores = simulate.simulate_slate(proj, dep, self.num_simulations)

        np.save(self.output().path, simulated_scores)
