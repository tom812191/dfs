import luigi
import pandas as pd
import os
import datetime
import numpy as np

import jobs.model.data as data
import jobs.model.optimize as optimize
import jobs.model.simulate as simulate
import util.report as report
import config as cfg


class ReportAll(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())

    def requires(self):
        return [
            CreateEntry(date=self.date),
            SimulatedPerformance(date=self.date),
            PlayerExposures(date=self.date),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PROJECTIONS,
                                              '{}_master.txt'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        with self.output().open('w') as f:
            print('DONE', file=f)


class CreateEntry(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())

    def requires(self):
        return [
            optimize.OptimizeLineupSet(date=self.date),
            data.DownloadProjectionData(),
            data.DKUploadTemplate(date=self.date)
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PROJECTIONS,
                                              '{}_entry.csv'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        entry = np.load(self.input()[0].path)

        with self.input()[1].open('r') as f:
            proj = pd.read_csv(f)

        proj = proj.sort_values(['TEAM', 'PLAYER']).reset_index()

        players = proj['PLAYER'].values
        entry_names = players[entry]

        with self.input()[2].open('r') as f:
            dk_template = pd.read_csv(f, skiprows=7)

        dk_template = dk_template.iloc[:, 8:12].reset_index(drop=True)
        dk_template.columns = ['POSITION', 'NAME_AND_ID', 'NAME', 'ID']

        columns = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        lineups = pd.DataFrame(columns=columns)

        for lineup in report.iter_entries(dk_template, entry_names):
            lineups = lineups.append(pd.Series(lineup, index=columns), ignore_index=True)

        with self.output().open('w') as f:
            lineups.to_csv(f, index=False)


class PlayerExposures(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())

    def requires(self):
        return [
            optimize.OptimizeLineupSet(date=self.date),
            data.DownloadProjectionData(),
            data.DKUploadTemplate(date=self.date)
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PROJECTIONS,
                                              '{}_exposures.csv'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        entry = np.load(self.input()[0].path)

        with self.input()[1].open('r') as f:
            proj = pd.read_csv(f)

        proj = proj.sort_values(['TEAM', 'PLAYER']).reset_index()

        players = proj['PLAYER'].values
        entry_names = players[entry]

        with self.input()[2].open('r') as f:
            dk_template = pd.read_csv(f, skiprows=7)

        dk_template = dk_template.iloc[:, 8:12].reset_index(drop=True)
        dk_template.columns = ['POSITION', 'NAME_AND_ID', 'NAME', 'ID']

        exposures = report.calculate_exposure(dk_template, entry_names)
        proj = proj[['PLAYER', 'SALARY', 'TEAM', 'OPP', 'CEIL', 'FLOOR', 'PROJECTION', 'PTS_DK_STD']].set_index('PLAYER')

        out = exposures.join(proj)
        out['PTS_PER_K'] = 1000 * out['PROJECTION'] / out['SALARY']

        with self.output().open('w') as f:
            out.to_csv(f, index=False)


class SimulatedPerformance(luigi.Task):
    date = luigi.DateParameter(default=datetime.date.today())
    target = luigi.IntParameter(default=330)

    def requires(self):
        return [
            optimize.OptimizeLineupSet(date=self.date),
            simulate.SimulateCrossValidationSlate(date=self.date),
        ]

    def output(self):
        return luigi.LocalTarget(os.path.join(cfg.DATA_PROJECTIONS,
                                              '{}_performance.npy'.format(self.date.strftime('%Y%m%d'))))

    def run(self):
        entry = np.load(self.input()[0].path)
        scores = np.load(self.input()[1].path)

        # Results will have the simulated scores with the following dimensions
        # Dim 0: Different lineups
        # Dim 1: Different players
        # Dim 2: Different sim
        # Value: player score in sim
        results = scores[entry]

        hit_target = np.sum(results, axis=1) > self.target
        pct = float(hit_target.any(axis=0).mean())

        print('----------------------------------------------')
        print('-----------------TEST RESULTS-----------------')
        print('----------------------------------------------')
        print('{}'.format(pct))

        np.save(self.output().path, np.array([pct]))
