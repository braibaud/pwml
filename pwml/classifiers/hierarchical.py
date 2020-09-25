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
import sklearn.preprocessing as skp
import sklearn.model_selection as skms
import sklearn.pipeline as skpl
import sklearn.decomposition as skd
import sklearn.linear_model as sklm
import sklearn.ensemble as skle
import sklearn.neighbors as skln
import sklearn.dummy as sky

from ..utilities import imagehelpers as ih
from ..utilities import httphelpers as hh
from ..utilities import filehelpers as fh
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

        self.X = None
        self.y = None

    def save_configuration(self):
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

        for child in self.children:
            configuration['children'].append(
                child.save_configuration())

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

        self.tune_params = {}
        self.tune_params['classifiers'] = {}

        if len(self.classes) > 1:

            # Create training and testing partition (validation
            # split is handled by the 'StratifiedKFold' object)
            X_tr, X_te, y_tr, y_te = skms.train_test_split(
                self.X, 
                self.y, 
                stratify=self.y,
                shuffle=True,
                random_state=0,
                test_size=test_size)

            print('    -> Current (train) size: "{0}"; Number of classes "{1}"; Stratified-Size "{2}".'.format(
                X_tr.shape[0],
                len(self.classes),
                m.floor(X_tr.shape[0] / min(X_tr.shape[0], 5))))

            while X_tr.shape[0] < 2*len(self.classes):
                print('    -> Adjusting training dataset size. Current size: "{0}"; Number of classes "{1}".'.format(
                    X_tr.shape[0],
                    len(self.classes)))

                X_tr = np.append(X_tr, X_tr.copy())
                y_tr = np.append(y_tr, y_tr.copy())

            while m.floor(X_tr.shape[0] / min(X_tr.shape[0], 5)) < 2*len(self.classes):
                print('    -> Adjusting training dataset size. Current size: "{0}"; Number of classes "{1}".'.format(
                    X_tr.shape[0],
                    len(self.classes)))

                X_tr = np.append(X_tr, X_tr.copy())
                y_tr = np.append(y_tr, y_tr.copy())

            while X_te.shape[0] < 2*len(self.classes):
                print('    -> Adjusting testing dataset size. Current size: "{0}"; Number of classes "{1}".'.format(
                    X_te.shape[0],
                    len(self.classes)))

                X_te = np.append(X_te, X_te.copy())
                y_te = np.append(y_te, y_te.copy())

            for classifier_name, classifier_value in classifiers.items():

                if X_tr.shape[0] > 10000 and classifier_name == 'KNeighborsClassifier':
                    print('    -> Skipping KNeighborsClassifier as training set too large ({0} > 10000)'.format(
                        X_tr.shape[0]))

                    continue

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
                    ('std', skp.StandardScaler())]

                if min_pca is not None:
                    if X_tr.shape[0] > X_tr.shape[1] and X_tr.shape[1] > min_pca:
                        steps.append(
                            ('pca', skd.PCA(
                                random_state=0)))

                        dict_grid['pca__n_components'] = HierarchyElement.get_pca_nb_components(min_pca, X_tr.shape[1], 3)

                if classifier_name == 'SGDClassifier':
                    steps.append(
                        (classifier_name, sklm.SGDClassifier(
                            max_iter=1000, 
                            shuffle=True,
                            random_state=0, 
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

                elif classifier_name == 'KNeighborsClassifier':
                    steps.append(
                        (classifier_name, skln.KNeighborsClassifier(
                            weights='distance',
                            n_jobs=2)))

                # Create a pipeline for the work to be done
                pipe = skpl.Pipeline(steps)

                for param_name, param_value in classifier_value.items():
                    # Add the search space to the grid
                    dict_grid['{0}__{1}'.format(classifier_name, param_name)] = param_value

                # create the k-fold object
                kfold = skms.StratifiedKFold(
                    n_splits=min(X_tr.shape[0], 5), 
                    random_state=0, 
                    shuffle=True)

                search = skms.GridSearchCV(
                    estimator=pipe,
                    param_grid=dict_grid,
                    cv=kfold,
                    n_jobs=2)

                # capture start time
                start_time = ti.time()

                search.fit(
                    X=X_tr, 
                    y=y_tr)

                elapsed_time = dt.timedelta(seconds=int(round(ti.time() - start_time)))

                # capture elapsed time
                self.tune_params['classifiers'][classifier_name]['fit_time'] = elapsed_time.total_seconds()

                # capture all tuning parameters
                self.tune_params['classifiers'][classifier_name]['parameters'].update(search.best_params_)

                # keep the best estimator
                self.tune_params['classifiers'][classifier_name]['best_estimator'] = search.best_estimator_

                # capture the scores
                self.tune_params['classifiers'][classifier_name]['results'] = {
                    'validation': search.best_score_,
                    'test': search.score(X_te, y_te)
                    }

                print('        -> Best validation score: {0:.4%}'.format(
                    self.tune_params['classifiers'][classifier_name]['results']['validation']))

                print('        -> Test score: {0:.4%}'.format(
                    self.tune_params['classifiers'][classifier_name]['results']['test']))

                print('        -> Tuning time: {0} ({1}s)'.format(elapsed_time, elapsed_time.total_seconds()))

                if self.tune_params['classifiers'][classifier_name]['results']['test'] > 0.99:
                    print('        -> Skipping other classifiers as test score > 99%')
                    break

        else:
            classifier_name = 'DummyRegressor'

            print('    -> Tuning "{0}"'.format(classifier_name))

            self.tune_params['classifiers'][classifier_name] = {}

            estimator = skpl.Pipeline(
                steps=[ (classifier_name, sky.DummyRegressor(strategy='constant', constant=0)) ])

            estimator.fit(
                X=self.X, 
                y=self.y)
            
            self.tune_params['classifiers'][classifier_name]['results'] = {
                'validation': None,
                'test': 1.0
                }

            self.tune_params['classifiers'][classifier_name]['fit_time'] = 0
            self.tune_params['classifiers'][classifier_name]['parameters'] = {}
            self.tune_params['classifiers'][classifier_name]['best_estimator'] = estimator

        # find the model with the best results.
        all_classifiers = list(self.tune_params['classifiers'].keys())
        all_results = [self.tune_params['classifiers'][classifier]['results']['test'] for classifier in all_classifiers]

        best_estimator_index = np.argmax(all_results)
        best_estimator_name = all_classifiers[best_estimator_index]

        self.estimator = self.tune_params['classifiers'][best_estimator_name]['best_estimator']

        self.tune_params['best_score'] = self.tune_params['classifiers'][best_estimator_name]['results']['test']
        self.tune_params['best_classifier'] = best_estimator_name

        print('    -> Best classifier is "{0}" with a test score of {1:.4%}'.format(
            best_estimator_name,
            self.tune_params['best_score']))

    def save_data(self, filename):
        fh.save_to_npz(
            filename,
            {
                'X': self.X,
                'y': self.y,
                'classes': self.classes
            })

        return filename

    def save_cache(self, filename):
        fh.save_to_npz(
            filename,
            {
                'X': self.X
            })

        return filename

    def load_data(self, filename):
        d = fh.load_from_npz(
                filename)

        self.X = d['X']
        self.y = d['y']
        self.classes = d['classes']

    def set_min_samples_per_class(self, min_samples_per_class=2):

        _X = []
        _y = []

        indices = np.arange(len(self.y))

        classes_values, samples_count = np.unique(self.y, return_counts=True)

        for classes_value in classes_values:
            if samples_count[classes_value] < min_samples_per_class:
                missing = min_samples_per_class - samples_count[classes_value]
                cycle = it.cycle(indices[(self.y == classes_value)])

                for _ in np.arange(missing):
                    new_idx = next(cycle)

                    _X.append(self.X[new_idx])
                    _y.append(self.y[new_idx])

        if len(_X) > 0:
            self.X = np.append(
                self.X,
                _X,
                axis=0)

            self.y = np.append(
                self.y,
                _y,
                axis=0)

    def get_dataset_balance_status(self):

        a, c = np.unique(
            self.y, 
            return_counts=True)

        df = pd.DataFrame.from_dict({
            'label_idx': a,
            'label_names': self.classes,
            'label_count': c
        })

        df.set_index('label_idx', inplace=True)
        return df


class HierarchicalClassifierModel(object):

    def __init__(self, model_name, experiment_name, input_features, output_feature_hierarchy, max_nb_categories=128):
        self.embedders = {}
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.input_features = input_features
        self.output_feature_hierarchy = output_feature_hierarchy
        self.max_nb_categories = max_nb_categories
        self.hierarchy = None

        self.register_default_embedders()

    def save_configuration(self):
        configuration = {
            'model_name': self.model_name,
            'input_features': self.input_features,
            'output_feature_hierarchy': self.output_feature_hierarchy,
            'max_nb_categories': self.max_nb_categories,
            'hierarchy': self.hierarchy.save_configuration(),
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

        model.hierarchy = HierarchyElement.load_configuration(configuration['hierarchy'])

        return model

    def save_data(self, filename):
        self.hierarchy.save_data(filename)

    def register_default_embedders(self):
        self.register_embedder(emb.CategoryEmbedder())
        self.register_embedder(emb.TextEmbedder())
        self.register_embedder(emb.NumericEmbedder())

    def save_model(self, filepath):
        model_filepath = fh.save_to_pickle(
            filepath=filepath,
            data=self.save_configuration())

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

    def load_from_csv(self, input_file, sep=',', header=0, cache={}):
        self.load_from_dataframe(
            data=pd.read_csv(
                filepath_or_buffer=input_file,
                sep=sep,
                header=header),
            cache=cache)

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

    def load_from_dataframe(self, data, cache={}):
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
            cache=cache,
            parent_output_feature=None,
            output_feature=self.output_feature_hierarchy,
            filter_value=None,
            depth=0,
            path='.')
        
    def _load_from_dataframe(self, data, cache, parent_output_feature, output_feature, filter_value, depth, path):
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

        _data = None

        if element.parent_output_feature is None:
            _data = data
        else:
            _data = data[(data[element.parent_output_feature.feature_name] == element.filter_value)]

        element.classes = _data[element.output_feature.feature_name].unique().tolist()

        print(' -> There are {0} classes and {1} samples.'.format(
            len(element.classes),
            _data.shape[0]))

        x = []
        y = []
        index = 0

        # load the data
        for r in _data.itertuples(index=True):
            
            input = r._asdict()
            
            if not self.valid_input(input):
                raise OverflowError(
                    'Invalid input (missing one or more features): "{0}".'.format(input))

            x.append(
                self._get_vector(input, input['Index'], cache))

            y.append(
                element.classes.index(input[element.output_feature.feature_name]))

            index = index + 1
        
            if index % 10000 == 0:
                print(' -> Processed {0} inputs so far.'.format(index))

        if len(x) > 0 and len(y) > 0:
            element.X = np.array(x, dtype=np.float64)
            element.y = np.array(y, dtype=np.int32)

            element.set_min_samples_per_class(
                min_samples_per_class=10)

            print(' -> There are {0} classes and {1} samples.'.format(
                len(element.classes),
                element.y.shape[0]))

        if element.output_feature.child_feature is not None:
            for class_value in element.classes:
                child_element = self._load_from_dataframe(
                    data=_data,
                    cache=cache,
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
                    proba_values = hierarchy_element.estimator.predict_proba([X])[0]

                    proba_results = {}

                    for idx, proba_value in enumerate(proba_values):
                        proba_results[hierarchy_element.get_class_name(idx)] = float(proba_value)

                    result['predicted_proba'] = proba_results

                output.append(result)

            if len(hierarchy_element.children) > 0:
                self._predict(
                    X=X,
                    hierarchy_element=hierarchy_element.get_child_by_filter_value(y_name),
                    output=output,
                    append_proba=append_proba)

    def _get_vector(self, input, index=None, cache=None):
        
        _x = HierarchicalClassifierModel.try_and_get_from_cache(index, cache)
        
        if _x is None:
            for input_feature in self.input_features:
                _vect = self.embedders[input_feature.feature_type].embed_data(
                    input[input_feature.feature_name],
                    input_feature)

                # append to the vector
                if _x is None:
                    _x = _vect.copy()
                else:
                    _x = np.append(_x, _vect)
            
            if index is not None and cache is not None:
                cache[index] = _x

        return _x

    @staticmethod
    def try_and_get_from_cache(index, cache):
        if cache is None:
            return None
        elif index is None:
            return None
        else:
            if type(cache).__name__ == 'dict':
                if index in cache:
                    return cache[index]
                else:
                    return None
            elif type(cache).__name__ == 'ndarray':
                if index < cache.shape[0]:
                    return cache[index]
                else:
                    return None
