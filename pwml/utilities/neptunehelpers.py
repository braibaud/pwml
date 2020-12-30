import logging
import os as os
import tempfile
import datetime as dt
from matplotlib import pyplot as plt

import neptune as npt

from neptunecontrib import api as npta
from neptunecontrib import hpo as npth
from neptunecontrib.monitoring import optuna as nmo
from neptunecontrib.monitoring import keras as nmk


class ExperimentManager(object):

    def __init__(self, log, project_name, experiment_name, experiment_params, experiment_tags):

        logging.getLogger('neptune').setLevel(logging.ERROR)

        self.log = log
        self.project = None
        self.experiment = None
        self._optuna_callback = None
        self._keras_callback = None
        self._experiment_name = None

        if self.log:
            self.project=npt.set_project(
                project_qualified_name=project_name)

            self.experiment = npt.create_experiment(
                name=experiment_name,
                params=experiment_params)

            for tag in experiment_tags:
                self.experiment.append_tags(tag)

    @property
    def experiment_name(self):
        if self._experiment_name is None:
            if self.log and self.experiment:
                self._experiment_name = self.experiment.id
            else:
                self._experiment_name = ExperimentManager.get_fake_name()

        return self._experiment_name

    @property
    def optuna_callback(self):
        if self.log and self._optuna_callback is None:
                self._optuna_callback = nmo.NeptuneCallback(
                    experiment=self.experiment)

        return self._optuna_callback

    @property
    def keras_callback(self):
        if self.log and self._keras_callback is None:
                self._keras_callback = nmk.NeptuneCallback(
                    experiment=self.experiment)

        return self._keras_callback

    @staticmethod
    def get_fake_name():
        return str(int(dt.datetime.now().timestamp()))

    def get_experiment_id(self):
        if self.log and self.experiment:
            return self.experiment.id
        else:
            return None

    def add_experiment_tags(self, tag):
        if self.log and self.experiment:
            self.experiment.append_tags(tag)

            print('[Tag] {0}'.format(tag))

    def set_experiment_property(self, key, value):
        if self.log and self.experiment:
            self.experiment.set_property(
                key=key, 
                value=value)

            print('[Property] {0}: {1}'.format(key, value))

    def log_data_to_neptune(self, data, name, log_index=False, log_to_table=True, log_to_artifact=True):
        if self.log and self.experiment:
            if name is None:
                name = ExperimentManager.get_fake_name()

            if log_to_table:
                
                npta.log_table(
                    name='{0}.csv'.format(name), 
                    table=data,
                    experiment=self.experiment)

            if log_to_artifact:
                
                file_path = os.path.join(
                    tempfile.gettempdir(),
                    '{0}.csv'.format(name))

                data.to_csv(file_path, index=log_index)

                self.experiment.log_artifact(
                    file_path,
                    'data/{0}.csv'.format(name))

    def log_chart_to_neptune(self, figure, name, log_to_artifact=True):
        if self.log and self.experiment:
            if name is None:
                name = ExperimentManager.get_fake_name()

            if log_to_artifact:
                file_path = os.path.join(
                    tempfile.gettempdir(),
                    '{0}.jpg'.format(name))

                figure.savefig(
                    fname=file_path, 
                    dpi=96,
                    format='jpg',
                    quality=90,
                    bbox_inches='tight')

                self.experiment.log_artifact(
                    file_path,
                    'charts/{0}.jpg'.format(name))
