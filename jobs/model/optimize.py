import luigi
import datetime
import os
import pandas as pd
import numpy as np
import pickle

from jobs.model import data, simulate, misc
from util.optimize import lp, ga, sa
import config


class GenerateLineupUniverse(luigi.Task):
    """
    Create a set of lineups from which our final lineup set will be selected
    """
    date = luigi.DateParameter(default=datetime.date.today())
    num_lineups_universe = luigi.IntParameter(default=20000)

    def requires(self):
        return [
            data.DownloadProjectionData(date=self.date),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_PROJECTIONS,
                                              '{}_lineups.npy'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        with self.input()[0].open('r') as f:
            proj = pd.read_csv(f)

        lineups = lp.generate_lineup_universe(proj, num_lineups=self.num_lineups_universe, progress_cb=self.progress_cb)

        np.save(self.output().path, lineups)

    def progress_cb(self, pct_complete):
        self.set_status_message('Progress: {}%'.format(round(pct_complete*100, 1)))


class OptimizeLineupSet(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())
    num_lineups_universe = luigi.IntParameter(default=20000)
    num_lineups_entry = luigi.IntParameter(default=100)
    num_simulations = luigi.IntParameter(default=50000)
    num_games = luigi.IntParameter(default=5)
    contest_size = luigi.IntParameter(default=51525)
    algorithm = luigi.Parameter(default='ga')

    def requires(self):
        return [
            GenerateLineupUniverse(date=self.date, num_lineups_universe=self.num_lineups_universe),
            simulate.SimulateSlate(date=self.date, num_simulations=self.num_simulations),
            misc.ScoreTargetLinearModel(),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(config.DATA_PROJECTIONS,
                                              '{}_entry.npy'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        lineup_universe = np.load(self.input()[0].path)
        simulated_scores = np.load(self.input()[1].path)

        with open(self.input()[2].path, 'rb') as f:
            target_model = pickle.load(f)

        target = target_model.predict(np.array([[self.contest_size, self.num_games]]))[0, 0]

        history = None

        if self.algorithm == 'ga':
            entry, _ = ga.optimize_lineup_set(lineup_universe, simulated_scores, num_lineups=self.num_lineups_entry,
                                              target=target)

        elif self.algorithm == 'sa':
            entry, _, history = sa.optimize_lineup_set(lineup_universe, simulated_scores,
                                                       num_lineups=self.num_lineups_entry, verbose=1, target=target,
                                                       store_history=True, end_temperature=0.0000001)

        elif self.algorithm == 'hybrid':
            entry, best_state = ga.optimize_lineup_set(lineup_universe, simulated_scores,
                                                       num_lineups=self.num_lineups_entry, target=target, ga_num_gen=75)

            entry, _, history = sa.optimize_lineup_set(lineup_universe, simulated_scores, initial_state=best_state,
                                                       num_lineups=self.num_lineups_entry, verbose=1, target=target,
                                                       store_history=True, initial_temperature=0.001,
                                                       end_temperature=0.0000001)

        else:
            entry, _ = ga.optimize_lineup_set(lineup_universe, simulated_scores)

        np.save(self.output().path, entry)

        if history is not None:
            np.save('/Users/tom/tmp/history.npy', np.array(history))
