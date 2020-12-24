import pandas as pd
import numpy as np
import matplotlib as mat
from matplotlib import pyplot as plt
import pylab as pyl
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

from ..utilities import commonhelpers as cmn
from ..utilities import graphichelpers as gph


def plot_time_series(title=None, subtitle=None, name=None,
    
    # Accepts an array of confidences
    confidence=None,
    confidence_lower_col='yhat_lower',
    confidence_upper_col='yhat_upper',
    confidence_label='95% interval',
    
    # Accepts an array of trainings
    training=None,
    training_col='y',
    training_label='Training data',
    
    # Accepts an array of testings
    testing=None,
    testing_col='y',
    testing_label='Testing data',
    
    # Accepts an array of predictions
    prediction=None,
    prediction_col='yhat',
    prediction_label='Predicted data',
    
    # Accepts an array of residuals
    residual=None,
    residual_col='residual',
    residual_label='Error',
    
    # Accepts an array of trends
    trend=None,
    trend_col='trend',
    trend_label='Trend',
    
    # Accepts an array of splits
    split=None,
    split_label='Train/test split',
    
    window_size=None,
    include_zero=True,
    experiment_manager=None):


    # syle
    gph.GraphicsStatics.initialize_matplotlib_styles()

    # Create the figure and the axes
    fig, ax = plt.subplots(
        figsize=gph.GraphicsStatics.g_landscape_fig_size)

    
    min_values = []
    max_values = []
    max_date = None
    
    if include_zero:
        min_values.append(0)
        max_values.append(0)
        
    ##### CONFIDENCE #####
    if confidence is not None:
        
        if type(confidence) is not list:
            confidence = [confidence]
        
        if type(confidence_lower_col) is not list:
            confidence_lower_col = [confidence_lower_col] * len(confidence)
        
        if type(confidence_upper_col) is not list:
            confidence_upper_col = [confidence_upper_col] * len(confidence)
        
        if type(confidence_label) is not list:
            confidence_label = [confidence_label] * len(confidence)
        
        for i_con, con in enumerate(confidence):

            ax.fill_between(
                x=con['ds'],
                y1=con[confidence_lower_col[i_con]],
                y2=con[confidence_upper_col[i_con]],
                color=gph.GraphicsStatics.get_color(6), 
                linestyle=gph.GraphicsStatics.get_linestyle(i_con),
                alpha=0.2,
                label=confidence_label[i_con])

            sns.lineplot(
                x='ds',
                y=confidence_lower_col[i_con],
                label=None,
                color=gph.GraphicsStatics.get_color(6), 
                linestyle=gph.GraphicsStatics.get_linestyle(i_con),
                alpha=0.8, 
                linewidth=1,
                data=con,
                ax=ax)

            sns.lineplot(
                x='ds',
                y=confidence_upper_col[i_con],
                label=None,
                color=gph.GraphicsStatics.get_color(6), 
                linestyle=gph.GraphicsStatics.get_linestyle(i_con),
                alpha=0.8, 
                linewidth=1,
                data=con,
                ax=ax)

            min_values.append(con[confidence_lower_col[i_con]].min())
            max_values.append(con[confidence_lower_col[i_con]].max())

            min_values.append(con[confidence_upper_col[i_con]].min())
            max_values.append(con[confidence_upper_col[i_con]].max())

            max_date = con['ds'].max()

        
    ##### RESIDUAL #####
    if residual is not None:
        
        if type(residual) is not list:
            residual = [residual]
        
        if type(residual_col) is not list:
            residual_col = [residual_col] * len(residual)
        
        if type(residual_label) is not list:
            residual_label = [residual_label] * len(residual)
            
        for i_res, res in enumerate(residual):
            
            ax.fill_between(
                x=res['ds'], 
                y1=0,
                y2=res[residual_col[i_res]], 
                color=gph.GraphicsStatics.get_color(4),
                linestyle=gph.GraphicsStatics.get_linestyle(i_res),
                alpha=0.2,
                label=residual_label[i_res])

            sns.lineplot(
                x='ds',
                y=residual_col[i_res],
                label=None,
                color=gph.GraphicsStatics.get_color(4), 
                linestyle=gph.GraphicsStatics.get_linestyle(i_res),
                alpha=0.8, 
                linewidth=1,
                data=res,
                ax=ax)

            min_values.append(0.0)
            max_values.append(0.0)

            min_values.append(res[residual_col[i_res]].min())
            max_values.append(res[residual_col[i_res]].max())

            max_date = res['ds'].max()
        

    ##### TRAINING #####
    if training is not None:
        
        if type(training) is not list:
            training = [training]
        
        if type(training_col) is not list:
            training_col = [training_col] * len(training)
        
        if type(training_label) is not list:
            training_label = [training_label] * len(training)
        
        for i_tra, tra in enumerate(training):

            sns.lineplot(
                x='ds',
                y=training_col[i_tra],
                label=training_label[i_tra],
                color=gph.GraphicsStatics.get_color(0), 
                linestyle=gph.GraphicsStatics.get_linestyle(i_tra),
                data=tra,
                ax=ax)

            min_values.append(tra[training_col[i_tra]].min())
            max_values.append(tra[training_col[i_tra]].max())

            max_date = tra['ds'].max()

            
    ##### TESTING #####
    if testing is not None:
        
        if type(testing) is not list:
            testing = [testing]
        
        if type(testing_col) is not list:
            testing_col = [testing_col] * len(testing)
        
        if type(testing_label) is not list:
            testing_label = [testing_label] * len(testing)
            
        for i_tes, tes in enumerate(testing):

            sns.lineplot(
                x='ds',
                y=testing_col[i_tes],
                label=testing_label[i_tes],
                color=gph.GraphicsStatics.get_color(1), 
                linestyle=gph.GraphicsStatics.get_linestyle(i_tes),
                data=tes,
                ax=ax)

            min_values.append(tes[testing_col[i_tes]].min())
            max_values.append(tes[testing_col[i_tes]].max())

            max_date = tes['ds'].max()

            
    ##### PREDICTION #####
    if prediction is not None:
        
        if type(prediction) is not list:
            prediction = [prediction]
        
        if type(prediction_col) is not list:
            prediction_col = [prediction_col] * len(prediction)
        
        if type(prediction_label) is not list:
            prediction_label = [prediction_label] * len(prediction)
            
        for i_pre, pre in enumerate(prediction):

            sns.lineplot(
                x='ds',
                y=prediction_col[i_pre],
                label=prediction_label[i_pre],
                color=gph.GraphicsStatics.get_color(2), 
                linestyle=gph.GraphicsStatics.get_linestyle(i_pre),
                data=pre,
                ax=ax)

            min_values.append(pre[prediction_col[i_pre]].min())
            max_values.append(pre[prediction_col[i_pre]].max())

            max_date = pre['ds'].max()

            
    ##### TREND #####
    if trend is not None:
        
        if type(trend) is not list:
            trend = [trend]
        
        if type(trend_col) is not list:
            trend_col = [trend_col] * len(trend)
        
        if type(trend_label) is not list:
            trend_label = [trend_label] * len(trend)
            
        for i_tre, tre in enumerate(trend):

            sns.lineplot(
                x='ds',
                y=trend_col[i_tre],
                label=trend_label[i_tre],
                color=gph.GraphicsStatics.get_color(7), 
                linestyle=gph.GraphicsStatics.get_linestyle(i_tre),
                data=tre,
                ax=ax)

            min_values.append(tre[trend_col[i_tre]].min())
            max_values.append(tre[trend_col[i_tre]].max())

            max_date = tre['ds'].max()

            
    ##### SPLIT #####
    if split is not None:
        
        if type(split) is not list:
            split = [split]
        
        if type(split_label) is not list:
            split_label = [split_label] * len(split)
            
        for i_spl, spl in enumerate(split):

            ax.axvline(
                x=spl,
                label=split_label[i_spl],
                color=gph.GraphicsStatics.get_color(3), 
                linestyle=gph.GraphicsStatics.get_linestyle(1), 
                linewidth=2)

    
    ##### ERROR AREA #####
    if training is not None and testing is not None and prediction is not None:
        
        for i, (tr, te, pr) in enumerate(zip(training, testing, prediction)):

            ax.fill_between(
                x=pd.concat([tr, te])['ds'], 
                y1=pd.concat([tr[training_col[i]], te[testing_col[i]]]),
                y2=pr[prediction_col[i]], 
                color=gph.GraphicsStatics.get_color(7),
                alpha=0.15)

            

    range_v = int((max(max_values) - min(min_values)) / 4)

    ax.set_ylim(
        bottom=min(min_values),
        top=max(max_values) + range_v)


    if window_size is not None and max_date is not None:
        
        min_date = max_date - window_size
        
        ax.set_xlim(
            left=min_date,
            right=max_date)


    ax.legend(
        loc='upper right')

    ax.set(
        xlabel='date', 
        ylabel='traffic')

    gph.GraphicsStatics.style_plot(
        title=title, 
        subtitle=subtitle, 
        tl_name=name,
        fig=fig,
        ax=ax,
        experiment_manager=experiment_manager)

    plt.show()

def plot_heatmap(data, title=None, subtitle=None, name=None, cbar_label='traffic', xlabel='fiscal week', ylabel='fiscal year', experiment_manager=None):

    # syle
    gph.GraphicsStatics.initialize_matplotlib_styles()

    # Create the figure and the axes
    fig, ax = plt.subplots(
        figsize=gph.GraphicsStatics.g_landscape_fig_size)

    sns.heatmap(
        data=data, 
        linewidths=.5,
        ax=ax,
        cmap='OrRd', 
        cbar_kws={
            'orientation': 'horizontal',
            'label': cbar_label
        })

    # plt.xlabel('fiscal week')

    ax.set(
        xlabel=xlabel, 
        ylabel=ylabel)

    gph.GraphicsStatics.style_plot(
        title=title, 
        subtitle=subtitle, 
        tl_name=name,
        fig=fig,
        ax=ax,
        experiment_manager=experiment_manager)

    plt.show()    

def plot_seasonal_decomposition(data, title=None, subtitle=None, name=None, period=None, xlabel='', ylabel='traffic', experiment_manager=None):

    decomposition_obj = seasonal_decompose(
        x=data, 
        model='additive',
        period=period,
        two_sided=True)

    # syle
    gph.GraphicsStatics.initialize_matplotlib_styles()

    # Create the figure and the axes
    fig, ax = plt.subplots(
        4, 1,
        figsize=gph.GraphicsStatics.g_square_fig_size)

    # Observed time series.
    decomposition_obj.observed.plot(
        ax=ax[0],
        c=gph.GraphicsStatics.get_color(0))
    
    ax[0].set(
        title='OBSERVED', 
        xlabel=xlabel, 
        ylabel=ylabel)

    # Trend component. 
    decomposition_obj.trend.plot(
        ax=ax[1],
        c=gph.GraphicsStatics.get_color(1))
    
    ax[1].set(
        title='TREND', 
        xlabel=xlabel, 
        ylabel=ylabel)

    # Seasonal component. 
    decomposition_obj.seasonal.plot(
        ax=ax[2],
        c=gph.GraphicsStatics.get_color(2))
    
    ax[2].set(
        title='SEASONAL', 
        xlabel=xlabel,
        ylabel=ylabel)

    # Residual.
    decomposition_obj.resid.plot(
        ax=ax[3],
        c=gph.GraphicsStatics.get_color(3))
    
    ax[3].set(
        title='RESIDUAL', 
        xlabel=xlabel, 
        ylabel=ylabel)

    gph.GraphicsStatics.style_plot(
        title=title, 
        subtitle=subtitle, 
        tl_name=name,
        fig=fig,
        ax=ax[0],
        top=0.87,
        tight_layout=True,
        experiment_manager=experiment_manager)

    plt.show()

def plot_autocorrelation(data, title=None, subtitle=None, name=None, lags=None, experiment_manager=None):

    # syle
    gph.GraphicsStatics.initialize_matplotlib_styles()

    # Create the figure and the axes
    fig, ax = plt.subplots(
        2, 1,
        figsize=gph.GraphicsStatics.g_square_fig_size)

    plot_acf(
        x=data,
        ax=ax[0],
        lags=lags,
        alpha=.05)

    plot_pacf(
        x=data,
        ax=ax[1],
        lags=lags,
        alpha=.05)

    gph.GraphicsStatics.style_plot(
        title=title, 
        subtitle=subtitle, 
        tl_name=name,
        fig=fig,
        ax=ax[0],
        top=0.87,
        tight_layout=True,
        experiment_manager=experiment_manager)

    plt.show()

def plot_time_series_dist(data, title=None, subtitle=None, name=None, bins=15, experiment_manager=None):

    # syle
    gph.GraphicsStatics.initialize_matplotlib_styles()

    # Create the figure and the axes
    fig, ax = plt.subplots(
        figsize=gph.GraphicsStatics.g_landscape_fig_size)
    
    data_mean = data.mean()
    data_std = data.std()

    sns.distplot(
        a=data, 
        ax=ax,
        bins=bins,
        rug=True)

    ax.axvline(
        x=data_mean,
        color=gph.GraphicsStatics.get_color(2),
        linestyle='--',
        label=r'$\mu$')

    ax.axvline(
        x=data_mean + 2*data_std,
        color=gph.GraphicsStatics.get_color(3),
        linestyle='--', 
        label=r'$\mu \pm 2\sigma$')

    ax.axvline(
        x=data_mean - 2*data_std,
        color=gph.GraphicsStatics.get_color(3),
        linestyle='--')

    ax.legend(
        loc='upper right')

    ax.set(
        xlabel='error', 
        ylabel='density')

    gph.GraphicsStatics.style_plot(
        title=title, 
        subtitle=subtitle, 
        tl_name=name,
        fig=fig,
        ax=ax,
        experiment_manager=experiment_manager)

    plt.show()

def plot_regression_comparison(title=None, subtitle=None, name=None, 
                               actual_train=None, actual_test=None, actual_col='y',
                               forecast_train=None, forecast_test=None, forecast_col='yhat',
                               experiment_manager=None):

    # syle
    gph.GraphicsStatics.initialize_matplotlib_styles()

    # Create the figure and the axes
    fig, ax = plt.subplots(
        figsize=gph.GraphicsStatics.g_landscape_fig_size)

    if actual_train is not None or actual_test is not None:
        
        start = None
        stop = None
        
        if actual_train is not None:
            start = actual_train[actual_col].min()
            stop = actual_train[actual_col].max()
        
        if actual_test is not None:
            if start is None:
                start = actual_test[actual_col].min()
            else:
                start = min([start, actual_test[actual_col].min()])

            if stop is None:
                stop = actual_test[actual_col].max()
            else:
                stop = max([stop, actual_test[actual_col].max()])
        
        # Generate diagonal line to plot. 
        d_x = np.linspace(
            start=start - 1, 
            stop=stop + 1,
            num=100)

        sns.lineplot(
            x=d_x, 
            y=d_x,
            dashes={'linestyle': '--'}, 
            color=gph.GraphicsStatics.get_color(3),
            ax=ax)

        ax.lines[0].set_linestyle('--')
    
    if actual_train is not None and forecast_train is not None:
        sns.regplot(
            x=actual_train[actual_col], 
            y=forecast_train[forecast_col], 
            color=gph.GraphicsStatics.get_color(0),
            label='Training data',
            line_kws={'lw': 2},
            ax=ax)

    if actual_test is not None and forecast_test is not None:
        sns.regplot(
            x=actual_test[actual_col], 
            y=forecast_test[forecast_col],
            color=gph.GraphicsStatics.get_color(1),
            label='Testing data',
            line_kws={'lw': 2},
            ax=ax)

    ax.legend(
        loc='lower right')

    ax.set(
        xlabel='actual', 
        ylabel='forecast')

    gph.GraphicsStatics.style_plot(
        title=title, 
        subtitle=subtitle, 
        tl_name=name,
        fig=fig,
        ax=ax,
        experiment_manager=experiment_manager)

    plt.show()

