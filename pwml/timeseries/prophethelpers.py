import pandas as pd
import numpy as np
import fbprophet as fbp
import matplotlib as mat
from matplotlib import pyplot as plt

from ..utilities import graphichelpers as gph


def regressor_index(m, name):
    """Given the name of a regressor, return its (column) index in the `beta` matrix.
    Parameters
    ----------
    m: Prophet model object, after fitting.
    name: Name of the regressor, as passed into the `add_regressor` function.
    Returns
    -------
    The column index of the regressor in the `beta` matrix.
    """
    return np.extract(
        m.train_component_cols[name] == 1, m.train_component_cols.index
    )[0]

def regressor_coefficients(m):
    """Summarise the coefficients of the extra regressors used in the model.
    For additive regressors, the coefficient represents the incremental impact
    on `y` of a unit increase in the regressor. For multiplicative regressors,
    the incremental impact is equal to `trend(t)` multiplied by the coefficient.
    Coefficients are measured on the original scale of the training data.
    Parameters
    ----------
    m: Prophet model object, after fitting.
    Returns
    -------
    pd.DataFrame containing:
    - `regressor`: Name of the regressor
    - `regressor_mode`: Whether the regressor has an additive or multiplicative
        effect on `y`.
    - `center`: The mean of the regressor if it was standardized. Otherwise 0.
    - `coef_lower`: Lower bound for the coefficient, estimated from the MCMC samples.
        Only different to `coef` if `mcmc_samples > 0`.
    - `coef`: Expected value of the coefficient.
    - `coef_upper`: Upper bound for the coefficient, estimated from MCMC samples.
        Only to different to `coef` if `mcmc_samples > 0`.
    """
    assert len(m.extra_regressors) > 0, 'No extra regressors found.'
    
    # trend_mean = m.params['trend'].mean()
    trend_std = m.params['trend'].std()
    
    coefs = []
    
    for regressor, params in m.extra_regressors.items():
        
        beta = m.params['beta'][:, regressor_index(m, regressor)]
        additive = (params['mode'] == 'additive')
        
        if params['mode'] == 'additive':
            coef = beta * m.y_scale / params['std']
        else:
            coef = beta / params['std']
            
        percentiles = [
            (1 - m.interval_width) / 2,
            1 - (1 - m.interval_width) / 2,
        ]
        
        coef_bounds = np.quantile(coef, q=percentiles)
        
        record = {
            'regressor': regressor,
            'regressor_mode': params['mode'],
            'center': float(params['mu']),
            'coef_lower': coef_bounds[0],
            'coef': np.mean(coef),
            'coef_upper': coef_bounds[1],
            'beta': float(beta[0]),
            'beta_abs': abs(float(beta[0])),
            'beta_lower': float(beta[0]) if additive else float(beta[0]) - trend_std,
            'beta_upper': float(beta[0]) if additive else float(beta[0]) + trend_std
        }
        
        coefs.append(record)

    return pd.DataFrame(coefs)

def plot_regressors_importance(m, title=None, subtitle=None, name=None, regressors_names=None, y_name='beta_abs', y_label='Importance %', experiment_manager=None, display=True):
    
    df = regressor_coefficients(m).sort_values(by=y_name, ascending=False)
    df = df[['regressor', y_name]]
    df = df.rename(columns={ y_name: 'importance' })
    df.loc[:, 'importance'] = df['importance']

    # syle
    gph.GraphicsStatics.initialize_matplotlib_styles()

    # Create the figure and the axes
    fig, ax = plt.subplots(
        figsize=gph.GraphicsStatics.g_landscape_fig_size)
    
    x = range(df.shape[0])
    y = list(np.array(df['importance']))
    
    rects = ax.bar(
        x=x,
        height=y)

    ax.set_xticks(x)

    labels = list(df['regressor'])
    
    if regressors_names is not None:
        labels = [regressors_names[label] for label in labels]
        
    ax.set_xticklabels(
        labels=labels)  

    ax.set(
        ylabel=y_label)

    for i, rect in enumerate(rects):
        
        height = rect.get_height()
        
        ax.annotate(
            '{0:.2%}'.format(y[i]),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom')

    gph.GraphicsStatics.style_plot(
        title=title, 
        subtitle=subtitle, 
        tl_name=name,
        fig=fig,
        ax=ax,
        experiment_manager=experiment_manager)

    if display:
        plt.show()
    else:
        return fig
