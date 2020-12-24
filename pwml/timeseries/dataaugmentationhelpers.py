import math as m
import pandas as pd
import numpy as np

from sklearn import preprocessing as skp
from sklearn import pipeline as skpl

from ..utilities import commonhelpers as cmn


def add_distributed_noise(ts, noise_std=0.05, noise_mean=0.0, noise_ratio=1.05, rng=None, random_seed=None):
    """Adds white noise to a time series 1D array.

    Args:
        ts (1D numpy array): time series data.
        noise_std (float, optional): standard deviation of the added noise. Defaults to 0.05.
        noise_mean (float, optional): mean of the added noise. Defaults to 0.0.
        noise_ratio (float, optional): ratio of the overall added noise. Defaults to 1.05.
        rng ([type], optional): random number generator. Will be created if None is passed. Defaults to None.
        random_seed ([type], optional): random seed to be used by the random generator. Defaults to None.

    Returns:
        (1D numpy array): same shape as `ts`. Time series with white noise added to it.
    """
    ts_mean = ts.mean()
    ts_std = ts.std()
    
    ts_scaled = (ts - ts_mean) / ts_std
    ts_length = len(ts_scaled)
    
    if rng is None:
        rng = np.random.default_rng(
            seed=random_seed)
    
    ts_scaled_noised = noise_ratio * (noise_std * rng.standard_normal(ts_length) + noise_mean) + ts_scaled
    
    return ts_std * ts_scaled_noised + ts_mean

def prepare_data(data, lags_in, cols_in, steps_in, cols_out, steps_out, augmentation_factor=0, noise_std=0.05, noise_mean=0.0, noise_ratio=1.05, verbose=1):
    """[summary]

    Args:
        data (DataFrame): [description]
        lags_in (list): lagged versions of input data to inject as input. If `None`, will be set to `[0]`. If a `list` is passed, value `0` is prepended to the list (and named `lags`).
        cols_in (list): list of columns to use as input.
        steps_in (int): number of input steps.
        cols_out (list): list of columns to use as output.
        steps_out (int): number of output steps.
        augmentation_factor (int, optional): Number of time the original dataset is replicated with noise addition. Defaults to 0.
        noise_std (float, optional): standard deviation of the added noise. Defaults to 0.05.
        noise_mean (float, optional): mean of the added noise. Defaults to 0.0.
        noise_ratio (float, optional): ratio of the overall added noise. Defaults to 1.05.
        verbose (int, optional): [description]. Defaults to 1.

    Raises:
        ValueError: in case there isn't enough data to produce a single sample.

    Returns:
        X (3D array of shape (samples * (augmentation_factor + 1), steps_in, len(cols_in) * len(lags))):
        y (3D array of shape (samples * (augmentation_factor + 1), steps_out, len(cols_out))) 
        index (list): the input data index.
        scaler_in (scikit-learn transformer): transformer used to fit and transform input data X.
        scaler_out (scikit-learn transformer): transformer used to fit and transform output data y.
        samples (int): the number of produced samples before data augmentation.
    """    

    lags = [0]

    if lags_in is not None:
        # Contains all lag values (even the value 0: no-lag)    
        lags = list(set(lags + lags_in))

    # Maximum lag value
    steps_lag = max(lags)

    # Input data feeds
    data_in = data[cols_in].values

    if verbose > 0:
        print('data_in.shape = {0}'.format(data_in.shape))
        print('data_in contains np.nan = {0}, count() = {1}'.format(
            np.isnan(data_in).max(),
            np.isnan(data_in).sum()))

    # Output data feeds
    data_out = data[cols_out].values

    if verbose > 0:
        print('data_out.shape = {0}'.format(data_out.shape))
        print('data_out contains np.nan = {0}, count() = {1}'.format(
            np.isnan(data_out).max(),
            np.isnan(data_out).sum()))

    # This is be the number of samples that can be built out of the input dataframe
    samples = data_in.shape[0] - (steps_in + steps_out + steps_lag) + 1

    if verbose > 0:
        print('samples = {0}'.format(samples))
    
    if samples < 1:
        raise ValueError('not enough data to produce 1 sample.')
    
    # Information to retain for return of function
    index = list(data.index)
    
    # Scaler for input data
    scaler_in = skpl.Pipeline([
        ('std', skp.StandardScaler()),
        ('minmax', skp.MinMaxScaler(feature_range=(0, 1)))])
    
    data_in_scaled = scaler_in.fit_transform(data_in)

    if verbose > 0:
        print('data_in_scaled.shape = {0}'.format(data_in_scaled.shape))
        print('data_in_scaled contains np.nan = {0}, count() = {1}'.format(
            np.isnan(data_in_scaled).max(),
            np.isnan(data_in_scaled).sum()))
    
    # Scaler for output data
    scaler_out = skpl.Pipeline([
        ('std', skp.StandardScaler()),
        ('minmax', skp.MinMaxScaler(feature_range=(0, 1)))])

    data_out_scaled = scaler_out.fit_transform(data_out)

    if verbose > 0:
        print('data_out_scaled.shape = {0}'.format(data_out_scaled.shape))
        print('data_out_scaled contains np.nan = {0}, count() = {1}'.format(
            np.isnan(data_out_scaled).max(),
            np.isnan(data_out_scaled).sum()))

    # Generated dataset   
    X = []
    y = []
    
    for sample in range(samples):

        for step_in in range(steps_in):

            for index_in in range(len(cols_in)):

                index_data_in_scaled = data_in_scaled[:, index_in]
                
                for lag in lags:

                    X.append(
                        index_data_in_scaled[sample + steps_lag + step_in - lag])

        for step_out in range(steps_out):

            for index_out in range(len(cols_out)):

                index_data_out_scaled =  data_out_scaled[:, index_out]

                y.append(
                    index_data_out_scaled[sample + steps_lag + steps_in + step_out])
    
    # Check if data augmentation is required
    if augmentation_factor is None:
        augmentation_factor = 0

    if augmentation_factor > 0:

        for _ in range(augmentation_factor):

            for sample in range(samples):

                for step_in in range(steps_in):

                    for index_in in range(len(cols_in)):

                        index_data_in_scaled = add_distributed_noise(
                            ts=data_in_scaled[:, index_in],
                            noise_std=noise_std,
                            noise_mean=noise_mean,
                            noise_ratio=noise_ratio)
                    
                        for lag in lags:

                            X.append(
                                index_data_in_scaled[sample + steps_lag + step_in - lag])

                for step_out in range(steps_out):

                    for index_out in range(len(cols_out)):

                        index_data_out_scaled =  add_distributed_noise(
                            ts=data_out_scaled[:, index_out],
                            noise_std=noise_std,
                            noise_mean=noise_mean,
                            noise_ratio=noise_ratio)

                        y.append(
                            index_data_out_scaled[sample + steps_lag + steps_in + step_out])

    X = np.array(X).reshape(
        samples * (augmentation_factor + 1), 
        steps_in, 
        len(cols_in) * len(lags))

    if verbose > 0:
        print('X.shape = {0}'.format(X.shape))
        print('X contains np.nan = {0}, count() = {1}'.format(
            np.isnan(X).max(),
            np.isnan(X).sum()))

    y = np.array(y).reshape(
        samples * (augmentation_factor + 1), 
        steps_out,
        len(cols_out))

    if verbose > 0:
        print('y.shape = {0}'.format(y.shape))
        print('y contains np.nan = {0}, count() = {1}'.format(
            np.isnan(y).max(),
            np.isnan(y).sum()))
    
    return X, y, index, scaler_in, scaler_out, samples