import os as os
import io as io
import math as m
import pandas as pd
import numpy as np
import copy as cp
import pickle as pk
import datetime as dt
import time as ti
import itertools as it
import sklearn as sk

import sklearn as sk
from sklearn import preprocessing as skp
from sklearn import model_selection as skms
from sklearn import pipeline as skpl
from sklearn import decomposition as skd
from sklearn import linear_model as sklm
from sklearn import ensemble as skle
from sklearn import neighbors as skln
from sklearn import dummy as sky
from sklearn import metrics as skm
from sklearn import calibration as skc
from sklearn.utils import validation as skuv

from ..utilities import imagehelpers as ih
from ..utilities import httphelpers as hh
from ..utilities import filehelpers as fh
from ..utilities import classificationhelpers as ch
from . import embedders as emb


class HierarchyElement(object):

    def __init__(self):

        self.path = None
        self.depth = None
        self.classes = None
        self.filter_value = None
        self.parent_output_feature = None
        self.output_feature = None
        self.tune_params = {}
        self.estimator = None
        self.children = []
        self.data = {}

    @property
    def X(self):
        return self.data['training']['X']

    @X.setter
    def X(self, value):
        self.data['training']['X'] = value

    @property
    def y(self):
        return self.data['training']['y']

    @y.setter
    def y(self, value):
        self.data['training']['y'] = value

    def prepare_data_subset(self, subset):
        if subset not in self.data:
            self.data[subset] = {
                'X': None,
                'y': None
            }

    def subset_has_data(self, subset):
        if subset not in self.data:
            return False

        if 'X' not in self.data[subset] or self.data[subset]['X'] is None:
            return False
        
        if 'y' not in self.data[subset] or self.data[subset]['y'] is None:
            return False

        return True

    def save_configuration(self, include_data=False):
        configuration = {
            'path': self.path,
            'depth': self.depth,
            'classes': self.classes,
            'filter_value': self.filter_value,
            'parent_output_feature': self.parent_output_feature,
            'output_feature': self.output_feature,
            'tune_params': self.tune_params,
            'estimator': self.estimator,
            'children': []
        }

        if include_data:
            configuration['data'] = self.data

        for child in self.children:
            configuration['children'].append(
                child.save_configuration(
                    include_data=include_data))

        return configuration

    @staticmethod
    def load_configuration(configuration):
        element = HierarchyElement()
        element.path = configuration['path']
        element.depth = configuration['depth']
        element.classes = configuration['classes']
        element.filter_value = configuration['filter_value']
        element.parent_output_feature = configuration['parent_output_feature']
        element.output_feature = configuration['output_feature']
        element.tune_params = configuration['tune_params']
        element.estimator = configuration['estimator']

        if 'data' in configuration:
            element.data = configuration['data']

        for child in configuration['children']:
            element.children.append(
                HierarchyElement.load_configuration(child))

        return element

    def get_child_by_filter_value(self, filter_value):
        for child in self.children:
            if child.filter_value == filter_value:
                return child
        return None

    def get_class_name(self, y):
        return self.classes[y]

    def tune_classifiers(self, test_size, classifiers, min_pca, search_n_jobs):

        self._tune_classifiers(
            test_size=test_size,
            classifiers=classifiers,
            min_pca=min_pca,
            search_n_jobs=search_n_jobs)

        for child in self.children:
            child.tune_classifiers(
                test_size=test_size,
                classifiers=classifiers,
                min_pca=min_pca,
                search_n_jobs=search_n_jobs)

    def score(self, X, y):
        if self.estimator is None:
            return None

        return self.estimator.score(X, y)

    def predict(self, X):
        if self.estimator is None:
            return None

        return self.estimator.predict(X)

    def get_accuracy(self):
        accuracy = self._get_accuracy()

        if len(self.children) > 0:
            children_weights = 0
            children_accuracy = 0

            for element in self.children:
                children_weights += element.get_weight()
                children_accuracy += element.get_accuracy() * element.get_weight()

            accuracy *= children_accuracy / children_weights

        return accuracy

    def _get_accuracy(self):
        return self.tune_params['best_score']

    def get_weight(self):
        return self.X.shape[0]

    def get_summary(self, show_parameters, output=None):
        if output is None:
            output = []

        output.append(
            self._get_summary(
                show_parameters=show_parameters))

        for element in self.children:
            element.get_summary(
                show_parameters=show_parameters,
                output=output)

        return output

    def _get_summary(self, show_parameters):
        d = {
            'path': self.path,
            'depth': self.depth,
            'nb_classes': len(self.classes),
            'nb_samples': self.X.shape[0],
            'best_classifier': self.tune_params['best_classifier'],
            'best_score': self.tune_params['best_score'],
            'fit_time': self.tune_params['classifiers'][self.tune_params['best_classifier']]['fit_time']
        }

        if show_parameters:
            d.update(
                self.tune_params['classifiers'][self.tune_params['best_classifier']]['parameters'])

        return d

    @staticmethod
    def get_pca_nb_components(min_features, max_features, max_length):

        output = []
        value = min_features

        while value < max_features and len(output) < max_length:
            output.append(value)
            value = 2*value

        if len(output) < max_length:
            output.append(max_features)

        return output

    def _tune_classifiers(
            self,
            test_size,
            classifiers,
            min_pca,
            search_n_jobs):

        print('Tuning classifiers for output-feature "{0}" for path "{1}".'.format(
            self.output_feature.feature_name,
            self.path))

        if not self.subset_has_data('training'):
            print('    -> Missing training data...')
            raise KeyError('Missing training data')

        self.tune_params = {}
        self.tune_params['classifiers'] = {}

        if self.subset_has_data('testing'):
            X_tr = self.X
            y_tr = self.y
            X_te = self.data['testing']['X']
            y_te = self.data['testing']['y']
        else:
            # Create training and testing partition (validation
            # split is handled by the 'StratifiedKFold' object)
            X_tr, X_te, y_tr, y_te = skms.train_test_split(
                self.X,
                self.y,
                stratify=self.y,
                shuffle=True,
                random_state=0,
                test_size=test_size)

        print('    -> Train size: ({0}); Test size: ({1}) Number of classes ({2}).'.format(
            X_tr.shape[0],
            X_te.shape[0],
            len(self.classes)))

        if len(self.classes) > 1:

            for classifier_name, classifier_value in classifiers.items():

                print('    -> Tuning "{0}"'.format(classifier_name))

                # Search grid
                dict_grid = {}

                self.tune_params['classifiers'][classifier_name] = {
                    'results': {},
                    'parameters': {},
                    'best_estimator': None
                }

                # Add standard scaller
                steps = [
                    ('std', skp.StandardScaler())
                ]

                if min_pca is not None and X_tr.shape[0] > X_tr.shape[1] and X_tr.shape[1] > min_pca:
                    steps.append(
                        ('pca', skd.PCA(
                            random_state=0)))

                    dict_grid['pca__n_components'] = HierarchyElement.get_pca_nb_components(
                        min_pca, X_tr.shape[1], 3)

                if classifier_name == 'SGDClassifier':
                    steps.append(
                        (classifier_name, sklm.SGDClassifier(
                            shuffle=True,
                            random_state=0,
                            max_iter=1000,
                            penalty='l2',
                            loss='log',
                            class_weight='balanced',
                            n_jobs=2)))

                elif classifier_name == 'RandomForestClassifier':
                    steps.append(
                        (classifier_name, skle.RandomForestClassifier(
                            random_state=0,
                            max_depth=None,
                            class_weight='balanced',
                            n_jobs=2)))

                # Create a pipeline for the work to be done
                pipe = skpl.Pipeline(steps)

                for param_name, param_value in classifier_value.items():
                    # Add the search space to the grid
                    dict_grid['{0}__{1}'.format(
                        classifier_name, param_name)] = param_value

                # create the k-fold object
                kfold = skms.StratifiedKFold(
                    n_splits=5,
                    random_state=0,
                    shuffle=True)

                search = skms.GridSearchCV(
                    estimator=pipe,
                    param_grid=dict_grid,
                    scoring='f1_weighted',
                    refit=True,
                    cv=kfold,
                    n_jobs=2)

                # capture start time
                start_time = ti.time()

                search.fit(
                    X=X_tr,
                    y=y_tr)

                elapsed_time = dt.timedelta(
                    seconds=int(round(ti.time() - start_time)))

                # capture elapsed time
                self.tune_params['classifiers'][classifier_name]['fit_time'] = elapsed_time.total_seconds()

                # capture all tuning parameters
                self.tune_params['classifiers'][classifier_name]['parameters'].update(search.best_params_)

                # keep the best estimator
                self.tune_params['classifiers'][classifier_name]['best_estimator'] = search.best_estimator_

                # capture the scores
                self.tune_params['classifiers'][classifier_name]['results'] = {
                    'validation': search.best_score_,
                    'test': search.score(
                        X=X_te, 
                        y=y_te)
                }

                print('        -> Best validation score: {0:.4%}'.format(
                    self.tune_params['classifiers'][classifier_name]['results']['validation']))

                print('        -> Test score: {0:.4%}'.format(
                    self.tune_params['classifiers'][classifier_name]['results']['test']))

                print(
                    '        -> Tuning time: {0} ({1}s)'.format(elapsed_time, elapsed_time.total_seconds()))

        else:
            classifier_name = 'DummyRegressor'

            print('    -> Tuning "{0}"'.format(classifier_name))

            self.tune_params['classifiers'][classifier_name] = {}

            estimator = skpl.Pipeline(
                steps=[
                    (classifier_name, sky.DummyRegressor(
                        strategy='constant', 
                        constant=0))
                ])

            estimator.fit(
                X=X_tr,
                y=y_tr)

            self.tune_params['classifiers'][classifier_name]['results'] = {
                'validation': 1.0,
                'test': 1.0
            }

            self.tune_params['classifiers'][classifier_name]['fit_time'] = 0
            self.tune_params['classifiers'][classifier_name]['parameters'] = {}
            self.tune_params['classifiers'][classifier_name]['best_estimator'] = estimator

        # find the model with the best results.
        all_classifiers = list(self.tune_params['classifiers'].keys())
        all_results = [self.tune_params['classifiers'][classifier]
                       ['results']['test'] for classifier in all_classifiers]

        best_estimator_index = np.argmax(all_results)
        best_estimator_name = all_classifiers[best_estimator_index]

        best_estimator = self.tune_params['classifiers'][best_estimator_name]['best_estimator']

        # We need to check whether the best estimator implements the 'predict_proba' method.
        # If it does, we can 1) calibrate the estimator and 2) compute the optimized thresholds. 
        if ch.MulticlassClassifierOptimizer.optimizable_model(best_estimator):

            print('    -> Optimizing "{0}"'.format(classifier_name))

            # Create a calibrated estimator
            optimized_estimator = ch.MulticlassClassifierOptimizer(
                model=best_estimator,
                classes=self.classes,
                scoring_function=ch.BinaryClassifierHelper.f1_score)

            self.estimator = optimized_estimator.fit(
                X=X_tr,
                y=y_tr)

            self.tune_params['classifiers'][best_estimator_name]['results']['train_optimized'] = optimized_estimator.score(
                X=X_tr,
                y=y_tr)

            self.tune_params['classifiers'][best_estimator_name]['results']['test_optimized'] = optimized_estimator.score(
                X=X_te,
                y=y_te)
        else:
            self.estimator = self.tune_params['classifiers'][best_estimator_name]['best_estimator']

        self.tune_params['best_score'] = self.tune_params['classifiers'][best_estimator_name]['results']['test']
        self.tune_params['best_classifier'] = best_estimator_name

        print('    -> Best classifier is "{0}" with a test score of {1:.4%}'.format(
            best_estimator_name,
            self.tune_params['best_score']))

        if 'test_optimized' in self.tune_params['classifiers'][best_estimator_name]['results']:

            print('    -> Optimized test score: {0:.4%}'.format(
                self.tune_params['classifiers'][best_estimator_name]['results']['test_optimized']))

    def plot_curve(self, class_name, X, y, n_bins=10):

        if isinstance(self.estimator, ch.MulticlassClassifierOptimizer):
            self.estimator.plot_curve(
                class_name=class_name,
                X=X,
                y=y,
                n_bins=n_bins)

    def plot_curves(self, X, y, n_bins=10, n_top=5):

        if isinstance(self.estimator, ch.MulticlassClassifierOptimizer):
            self.estimator.plot_curves(
                X=X,
                y=y,
                n_bins=n_bins,
                n_top=n_top)

    def get_metrics_by_class(self, X, y):

        if isinstance(self.estimator, ch.MulticlassClassifierOptimizer):
            return self.estimator.get_metrics_by_class(
                X=X,
                y=y)

    def save_data(self, filename):
        fh.save_to_npz(
            filename,
            {
                'data': self.data,
                'classes': self.classes
            })

        return filename

    def load_data(self, filename):
        d = fh.load_from_npz(
            filename)

        self.data = d['data']
        self.classes = d['classes']

    def get_min_samples_per_class(self, min_samples_per_class=10):

        for subset in self.data:
            if self.subset_has_data(subset):
                indices = np.arange(len(self.data[subset]['y']))

                for i, name in enumerate(self.classes):
                    c_indices = indices[(self.data[subset]['y'] == i)]

                    if len(c_indices) < min_samples_per_class:
                        yield {
                            'path': self.path,
                            'subset': subset,
                            'class': name,
                            'n_classes': len(c_indices)
                        }

        for child in self.children:
            for result in child.get_min_samples_per_class(
                min_samples_per_class=min_samples_per_class):
                yield result

    def set_min_samples_per_class(self, min_samples_per_class=10):

        for subset in self.data:

            if self.subset_has_data(subset):

                _X = []
                _y = []

                indices = np.arange(len(self.data[subset]['y']))

                for i, _ in enumerate(self.classes):

                    c_indices = indices[(self.data[subset]['y'] == i)]

                    if len(c_indices) > 0:
                        samples_count = c_indices.shape[0]

                        if samples_count < min_samples_per_class:
                            missing = min_samples_per_class - samples_count
                            cycle = it.cycle(c_indices)

                            for _ in np.arange(missing):
                                new_idx = next(cycle)

                                _X.append(self.data[subset]['X'][new_idx])
                                _y.append(self.data[subset]['y'][new_idx])

                if len(_X) > 0:
                    self.data[subset]['X'] = np.append(
                        self.data[subset]['X'],
                        _X,
                        axis=0)

                    self.data[subset]['y'] = np.append(
                        self.data[subset]['y'],
                        _y,
                        axis=0)

    def get_dataset_balance_status(self, subset='training'):

        df = None

        if self.subset_has_data(subset):

            a, c = np.unique(
                self.data[subset]['y'],
                return_counts=True)

            df = pd.DataFrame.from_dict({
                'label_idx': a,
                'label_names': self.classes,
                'label_count': c
            })

            df.set_index('label_idx', inplace=True)

        return df


class HierarchicalClassifierModel(object):

    def __init__(self, model_name, experiment_name, input_features, output_feature_hierarchy, cache_path=None, max_nb_categories=128):
        self.embedders = {}
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.input_features = input_features
        self.output_feature_hierarchy = output_feature_hierarchy
        self.cache_path = cache_path
        self.max_nb_categories = max_nb_categories
        self.hierarchy = None

        self.register_default_embedders(
            cache_path=cache_path)

    def save_configuration(self, include_data=False):
        configuration = {
            'model_name': self.model_name,
            'input_features': self.input_features,
            'output_feature_hierarchy': self.output_feature_hierarchy,
            'max_nb_categories': self.max_nb_categories,
            'hierarchy': self.hierarchy.save_configuration(
                include_data=include_data),
            'experiment_name': self.experiment_name
        }

        return configuration

    @staticmethod
    def load_configuration(configuration):
        model = HierarchicalClassifierModel(
            model_name=configuration['model_name'],
            input_features=configuration['input_features'],
            output_feature_hierarchy=configuration['output_feature_hierarchy'],
            experiment_name=configuration['experiment_name'],
            max_nb_categories=configuration['max_nb_categories'])

        model.hierarchy = HierarchyElement.load_configuration(
            configuration['hierarchy'])

        return model

    def save_data(self, filename):
        self.hierarchy.save_data(filename)

    def save_model(self, filepath, include_data=False):
        model_filepath = fh.save_to_pickle(
            filepath=filepath,
            data=self.save_configuration(
                include_data=include_data))

        print('Model saved in "{0}"'.format(model_filepath))

    @staticmethod
    def load_model(filepath):
        return HierarchicalClassifierModel.load_configuration(
            fh.load_from_pickle(
                filepath=filepath))

    def get_accuracy(self):
        if self.hierarchy is None:
            return None

        return self.hierarchy.get_accuracy()

    def get_dataset_balance_status(self):
        if self.hierarchy is None:
            return None

        return self.hierarchy.get_dataset_balance_status()

    def get_summary(self, show_parameters=False):
        if self.hierarchy is None:
            return None

        return pd.DataFrame(
            self.hierarchy.get_summary(
                show_parameters=show_parameters)).sort_values(
                    by=['depth', 'path'],
                    ascending=[True, True])

    def get_min_samples_per_class(self, min_samples_per_class=10):
        return pd.DataFrame.from_records(
            data=self.hierarchy.get_min_samples_per_class(
                min_samples_per_class=min_samples_per_class))

    def tune_classifiers(
        self,
        test_size=.2,
        min_pca=256,
        search_n_jobs=2,
        classifiers={
            'SGDClassifier': {
                'alpha': np.logspace(
                    start=-8,
                    stop=-1,
                    num=8)
            },
            'RandomForestClassifier': {
                'n_estimators': np.linspace(
                    start=64,
                    stop=256,
                    num=4,
                    dtype=np.int32)
            },
            'KNeighborsClassifier': {
                'n_neighbors': [4, 6, 8, 10]
            }}):

        if self.hierarchy is None:
            return None

        self.hierarchy.tune_classifiers(
            test_size=test_size,
            classifiers=classifiers,
            min_pca=min_pca,
            search_n_jobs=search_n_jobs)

    def register_embedder(self, embedder):
        self.embedders[embedder.input_type] = embedder

    def register_default_embedders(self, cache_path):
        self.register_embedder(emb.CategoryEmbedder(cache_path=cache_path))
        self.register_embedder(emb.NumericEmbedder(cache_path=cache_path))
        self.register_embedder(emb.TextEmbedder(cache_path=cache_path, **emb.NNLM_EN_DIM128))
        self.register_embedder(emb.TextEmbedder(cache_path=cache_path, **emb.UNIVERSAL_SENTENCE_ENCODER_LARGE))

    def flush_embedders_cache(self):
        for _, value in self.embedders.items():
            value.flush_cache()

    def load_from_csv(self, input_file, sep=',', header=0):
        self.load_from_dataframe(
            data=pd.read_csv(
                filepath_or_buffer=input_file,
                sep=sep,
                header=header))

    @staticmethod
    def analyze_csv(input_file, sep=',', header=0):
        return HierarchicalClassifierModel.analyze_dataframe(
            data=pd.read_csv(
                filepath_or_buffer=input_file,
                sep=sep,
                header=header))

    @staticmethod
    def analyze_dataframe(data):
        return data.describe(include=['object']).transpose()

    def process_data(self, data):

        X = []
        y = []
        index = 0

        y_cols = []
        output_feature = self.output_feature_hierarchy

        while output_feature is not None:
            y_cols.append(output_feature.feature_name)
            output_feature = output_feature.child_feature

        # load the data
        for r in data.itertuples(index=True):

            r_dict = r._asdict()

            X.append(
                self._get_vector(r_dict))

            _y = []

            for y_col in y_cols:
                _y.append(
                   r_dict[y_col])

            y.append(
                np.array(_y))

            index = index + 1

            if index % 10000 == 0:
                print(' -> Processed {0} inputs so far.'.format(index))

        return np.array(X, dtype=float), np.array(y, dtype=str)

    def load_from_dataframe(self, data, subset='training'):

        if self.hierarchy is None or subset == 'training':

            for input_feature in self.input_features:
                if input_feature.feature_type == 'category':
                    input_feature.classes = data[input_feature.feature_name].unique().tolist()

                    if len(input_feature.classes) > self.max_nb_categories:
                        raise OverflowError(
                            'Too many values ({0}) for category "{1}". Maximum number of categories allowed is {2}. Switch type to "text" instead.'.format(
                                len(input_feature.classes),
                                input_feature.feature_name,
                                self.max_nb_categories))

            self.hierarchy = self._load_from_dataframe(
                data=data,
                parent_output_feature=None,
                output_feature=self.output_feature_hierarchy,
                filter_value=None,
                depth=0,
                path='.')

        else:
            # Load additional data
            self._dataload_from_dataframe(
                element=self.hierarchy,
                data=data,
                subset=subset)

    def _dataload_from_dataframe(self, element, data, subset='training'):

        print('Processing output-feature "{0}" for path "{1}" for subset "{2}".'.format(
            element.output_feature.feature_name,
            element.path,
            subset))

        element.prepare_data_subset(
            subset=subset)
        
        _data = None

        if element.parent_output_feature is None:
            _data = data
        else:
            _data = data[
                (data[element.parent_output_feature.feature_name] == element.filter_value)
            ]

        x = []
        y = []
        index = 0

        # load the data
        for r in _data.itertuples(index=True):

            r_dict = r._asdict()

            if not self.valid_input(r_dict):
                raise OverflowError(
                    'Invalid input (missing one or more features): "{0}".'.format(r_dict))

            if r_dict[element.output_feature.feature_name] in element.classes:
                x.append(self._get_vector(r_dict))
                y.append(element.classes.index(r_dict[element.output_feature.feature_name]))
            else:
                print('  -> Unknown class: {0}'.format(r_dict[element.output_feature.feature_name]))

            index = index + 1

            if index % 10000 == 0:
                print(' -> Processed {0} inputs so far.'.format(index))

        if len(x) > 0 and len(y) > 0:

            element.data[subset]['X'] = np.array(x, dtype=np.float64)
            element.data[subset]['y'] = np.array(y, dtype=np.int32)

            # element.set_min_samples_per_class(
            #     min_samples_per_class=25)

            print(' -> There are {0} classes and {1} samples.'.format(
                len(element.classes),
                element.data[subset]['y'].shape[0]))

        for child_element in element.children:

            self._dataload_from_dataframe(
                element=child_element,
                data=_data,
                subset=subset)

    def _load_from_dataframe(self, data, parent_output_feature, output_feature, filter_value, depth, path):

        element = HierarchyElement()
        element.parent_output_feature = parent_output_feature
        element.output_feature = output_feature
        element.filter_value = filter_value
        element.depth = depth
        element.path = path

        print('Processing output-feature "{0}" for path "{1}".'.format(
            element.output_feature.feature_name,
            element.path))

        if element.output_feature is None:
            return None

        element.prepare_data_subset(
            subset='training')
        
        _data = None

        if element.parent_output_feature is None:
            _data = data
        else:
            _data = data[(
                data[element.parent_output_feature.feature_name] == element.filter_value)]

        element.classes = _data[element.output_feature.feature_name].unique(
        ).tolist()

        print(' -> There are {0} classes and {1} samples.'.format(
            len(element.classes),
            _data.shape[0]))

        x = []
        y = []
        index = 0

        # load the data
        for r in _data.itertuples(index=True):

            r_dict = r._asdict()

            if not self.valid_input(r_dict):
                raise OverflowError(
                    'Invalid input (missing one or more features): "{0}".'.format(r_dict))

            x.append(self._get_vector(r_dict))
            y.append(element.classes.index(r_dict[element.output_feature.feature_name]))

            index = index + 1

            if index % 10000 == 0:
                print(' -> Processed {0} inputs so far.'.format(index))

        if len(x) > 0 and len(y) > 0:
            element.X = np.array(x, dtype=np.float64)
            element.y = np.array(y, dtype=np.int32)

            element.set_min_samples_per_class(
                min_samples_per_class=25)

            print(' -> There are {0} classes and {1} samples.'.format(
                len(element.classes),
                element.y.shape[0]))

        if element.output_feature.child_feature is not None:
            for class_value in element.classes:
                child_element = self._load_from_dataframe(
                    data=_data,
                    parent_output_feature=element.output_feature,
                    output_feature=element.output_feature.child_feature,
                    filter_value=class_value,
                    depth=depth+1,
                    path='{0} / {1}'.format(element.path, class_value))

                if child_element is not None:
                    element.children.append(child_element)

        return element

    def valid_input(self, input):
        valid = True

        for input_feature in self.input_features:
            if input_feature.feature_name not in input:
                valid = False
                break

        return valid

    def predict(self, input, append_proba=False, last_level_only=False):
        output = []

        if self.valid_input(input):
            self._predict(
                X=self._get_vector(input),
                hierarchy_element=None,
                output=output,
                append_proba=append_proba,
                last_level_only=last_level_only)

        return output

    def _predict(self, X, hierarchy_element=None, output=None, append_proba=False, last_level_only=False):
        if hierarchy_element is None:
            if self.hierarchy is None:
                return None
            else:
                hierarchy_element = self.hierarchy

        if hierarchy_element.estimator is not None:
            if last_level_only == False or len(hierarchy_element.children) == 0:
                y = hierarchy_element.predict([X])[0]
                y_name = hierarchy_element.get_class_name(y)

                result = {
                    'output_feature': hierarchy_element.output_feature.feature_name,
                    'predicted_class': y_name,
                    'predicted_class_index': int(y)
                }

                if append_proba:
                    proba_values = hierarchy_element.estimator.predict_proba([X])[
                        0]

                    proba_results = {}

                    for idx, proba_value in enumerate(proba_values):
                        proba_results[hierarchy_element.get_class_name(
                            idx)] = float(proba_value)

                    result['predicted_proba'] = proba_results

                output.append(result)

            if len(hierarchy_element.children) > 0:
                self._predict(
                    X=X,
                    hierarchy_element=hierarchy_element.get_child_by_filter_value(
                        y_name),
                    output=output,
                    append_proba=append_proba)

    def _get_vector(self, input):

        _x = None

        for input_feature in self.input_features:
            _vect = self.embedders[input_feature.feature_type].embed_data(
                input[input_feature.feature_name],
                input_feature)

            # append to the vector
            if _x is None:
                _x = _vect.copy()
            else:
                _x = np.append(_x, _vect)

        return _x
