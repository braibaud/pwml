import math as m
import pandas as pd
import numpy as np

import matplotlib as mat
from matplotlib import pyplot as plt
import pylab as pyl
import seaborn as sns

from scipy import optimize as sco

import sklearn as sk
import sklearn.preprocessing as skp
import sklearn.model_selection as skms
import sklearn.pipeline as skpl
import sklearn.decomposition as skd
import sklearn.linear_model as sklm
import sklearn.ensemble as skle
import sklearn.neighbors as skln
import sklearn.dummy as sky
import sklearn.metrics as skm
import sklearn.calibration as skc

# Class holding static properties only
class GraphicsStatics(object):
    # Globals
    g_palette = None
    g_fig_size = (20, 10)
    g_square_fig_size = (20, 20)
    g_styles_initialized = False


def initialize_styles():

    if not GraphicsStatics.g_styles_initialized:
        sns.set()
        sns.set_style('whitegrid', {'axes.facecolor': '.9'})
        plt.style.use('fivethirtyeight')
        sns.set_context('talk')

        # Assign the default palette
        GraphicsStatics.g_palette = sns.color_palette('Set2', 8)

        sns.set_palette(
            GraphicsStatics.g_palette)

        pyl.rcParams['figure.figsize'] = GraphicsStatics.g_fig_size
        plt.rcParams['figure.figsize'] = GraphicsStatics.g_fig_size
        plt.rcParams['axes.labelsize'] = 18
        plt.rcParams['axes.titlesize'] = 18

        GraphicsStatics.g_styles_initialized = True

def plot_curves(title, best_threshold, y_true, y_score):

    # syle
    initialize_styles()

    roc_color = 'crimson'
    pr_color = 'royalblue'
    main_linestyle = 'solid'
    neutral_color = 'k'
    neutral_linestyle = 'dashed'
    lw = 2

    fig = plt.figure(
        figsize=(20, 30))

    ax = []
    
    gs = fig.add_gridspec(3, 2)
    
    ax.append(
        fig.add_subplot(gs[0, 0]))
    
    ax.append(
        fig.add_subplot(gs[0, 1]))
    
    ax.append(
        fig.add_subplot(gs[1, :]))
    
    ax.append(
        fig.add_subplot(gs[2, 0]))
    
    ax.append(
        fig.add_subplot(gs[2, 1]))

    fpr, tpr_recall, _ = skm.roc_curve(
        y_true, 
        y_score,
        pos_label=1)
    
    ax[0].step(
        fpr,
        tpr_recall,
        color=roc_color,
        alpha=0.2,
        where='post',
        linewidth=lw,
        linestyle=main_linestyle)
    
    ax[0].fill_between(
        fpr,
        tpr_recall,
        step='post',
        alpha=0.2,
        color=roc_color)
    
    # diagonal line
    ax[0].plot(
        [0, 1], 
        [0, 1], 
        color=neutral_color,
        lw=lw,
        alpha=0.6,
        linestyle=neutral_linestyle)
    
    ax[0].set_xlim([-0.1, 1.1])
    
    ax[0].set_ylim([-0.1, 1.1])
    
    ax[0].set_xlabel('FPR (Fall-Out)', fontweight='bold')
    
    ax[0].set_ylabel('TPR (Recall)', fontweight='bold')
    
    ax[0].set_title(
        'AUROC = {0:.2f}'.format(skm.auc(fpr, tpr_recall)),
        fontweight='bold')

    #Plot dots at certain decision thresholds, for clarity
    for d in [.1, .3, .5, .7, .9]:
        
        tpr_recall, fpr, _ = calculate_tpr_fpr_prec(
            threshold=d,
            y_true=y_true,
            y_score=y_score)
        
        ax[0].plot(
            fpr, 
            tpr_recall, 
            'o', 
            color=roc_color)
        
        ax[0].annotate(
            '  t = {0:.1f}  '.format(d), 
            (fpr, tpr_recall),
            ha='right',
            rotation=-45,
            size=16)

        
    precision, tpr_recall, _ = skm.precision_recall_curve(
        y_true,
        y_score)
        
    ax[1].step(
        tpr_recall,
        precision,
        color=pr_color,
        alpha=0.2,
        where='post',
        linewidth=lw,
        linestyle=main_linestyle)
        
    ax[1].fill_between(
        tpr_recall,
        precision,
        step='post',
        alpha=0.2,
        color=pr_color)
        
    ax[1].set_xlabel('TPR (Recall)', fontweight='bold')
        
    ax[1].set_ylabel('PPV (Precision)', fontweight='bold')
        
    ax[1].set_xlim([-0.1, 1.1])
        
    ax[1].set_ylim([-0.1, 1.1])
        
    ax[1].set_title(
        'Average Precision = {0:.2f}'.format(skm.average_precision_score(y_true, y_score)), 
        fontweight='bold')

    #Plot dots at certain decision thresholds, for clarity
    for d in [.1, .3, .5, .7, .9]:
        
        tpr_recall, _, precision = calculate_tpr_fpr_prec(
            threshold=d,
            y_true=y_true,
            y_score=y_score)
        
        ax[1].plot(
            tpr_recall, 
            precision, 
            'o',
            color=pr_color)
        
        ax[1].annotate(
            '  t = {0:.1f}  '.format(d), 
            (tpr_recall, precision),
            ha='left',
            rotation=45,
            size=16)

    # Recall / Precision: xing curves
    x_v = list(np.linspace(
        start=0.0, 
        stop=1.0, 
        num=200))
    
    tpr_v = []
    ppv_v = []
    
    for d in x_v:
        
        tpr, _, ppv = calculate_tpr_fpr_prec(
            threshold=d,
            y_true=y_true,
            y_score=y_score)
        
        tpr_v.append(tpr)
        ppv_v.append(ppv)
                
    ax[2].plot(x_v, tpr_v, label='TPR (Recall)')
    ax[2].plot(x_v, ppv_v, label='PPV (Precision)')


    if best_threshold is not None:
        
        x_opt = best_threshold

        tpr_opt, _, ppv_opt = calculate_tpr_fpr_prec(
            threshold=best_threshold,
            y_true=y_true,
            y_score=y_score)
        
        ax[2].plot(
            x_opt, 
            tpr_opt, 
            'o',
            color=pr_color)

        ax[2].plot(
            x_opt, 
            ppv_opt, 
            'o',
            color=pr_color)

        ax[2].plot(
            x_opt, 
            0, 
            'o',
            color=pr_color)
        
        ax[2].plot(
            [x_opt, x_opt], 
            [0, max(tpr_opt, ppv_opt)], 
            color=neutral_color,
            lw=lw,
            alpha=0.6,
            linestyle=neutral_linestyle)

        ax[2].annotate(
            '  t = {0:.4f}  '.format(x_opt), 
            (x_opt, 0),
            ha='left',
            rotation=45,
            size=16)
        
    ax[2].set_xlabel('Threshold', fontweight='bold')
    ax[2].set_ylabel('Metric', fontweight='bold')
    ax[2].set_xlim([-0.1, 1.1])
    ax[2].set_ylim([-0.1, 1.1])
    ax[2].legend(loc='upper center')
    
    ax[2].set_title(
        'Threshold Maximizing F1-Score = {0:.4f}'.format(x_opt), 
        fontweight='bold')

    n_bins = 25
    
    fraction_of_positives, mean_predicted_value = skc.calibration_curve(
        y_true=y_true,
        y_prob=y_score,
        n_bins=n_bins)
    
    # Reliability Curve (Calibration)
    ax[3].plot(
        [0, 1],
        [0, 1],
        'k:', 
        label='Perfectly calibrated')
    
    ax[3].plot(
        mean_predicted_value, 
        fraction_of_positives,
        'o-',
        color=GraphicsStatics.g_palette[1],
        label='Current')
    
    ax[3].set_xlabel('Mean predicted value', fontweight='bold')
    ax[3].set_ylabel('Fraction of positives', fontweight='bold')
    ax[3].set_ylim([-0.05, 1.05])
    ax[3].legend(loc='lower right')
    
    ax[3].set_title(
        'Reliability Curve (Calibration)', 
        fontweight='bold')

    
    # Probability Distribution
    ax[4].hist(
        y_score,
        range=(0, 1),
        log=True,
        bins=n_bins,
        histtype='bar',
        lw=2)    

    ax[4].set_xlabel('Mean predicted value', fontweight='bold')
    ax[4].set_ylabel('Count (log)', fontweight='bold')
    
    ax[4].set_title(
        'Probability Distribution', 
        fontweight='bold')
    
    
    plt.subplots_adjust(
        top=0.91,
        wspace=.2, 
        hspace=.2)
    
    fig.suptitle(
        t=title, 
        fontsize=26, 
        fontweight='bold',
        verticalalignment='top',
        horizontalalignment='center',
        fontstyle='italic',
        x=(fig.subplotpars.right + fig.subplotpars.left)/2)

    matrix = confusion_matrix_string(
        threshold=best_threshold,
        y_true=y_true,
        y_score=y_score)

    fig.text(
        x=0.02, 
        y=1, 
        s=matrix, 
        ha='left', 
        va='top',
        fontsize=14,
        fontstyle='italic',
        bbox={ 
            'facecolor': GraphicsStatics.g_palette[5], 
            'alpha': 0.5, 
            'pad': 7
        })

    metrics = metrics_string(
        threshold=best_threshold,
        y_true=y_true,
        y_score=y_score)
    
    fig.text(
        x=.95, 
        y=1, 
        s=metrics, 
        ha='right', 
        va='top',
        fontsize=14,
        fontstyle='italic',
        bbox={ 
            'facecolor': GraphicsStatics.g_palette[6], 
            'alpha': 0.5, 
            'pad': 7
        })    
    
        
    plt.show()

def get_metrics_by_class(y_true, y_score, y_pred, classes):
    
    records = []

    for i, class_name in enumerate(classes):

        record = {
            'Class': class_name
        }

        cm = skm.confusion_matrix(
            y_true=y_true[:, i], 
            y_pred=y_pred[:, i],
            labels=[0.0, 1.0])

        TN, FP, FN, TP = cm.ravel()

        record['TN'] = '{0}'.format(TN)
        record['FP'] = '{0}'.format(FP)
        record['FN'] = '{0}'.format(FN)
        record['TP'] = '{0}'.format(TP)

        record['Actuals Negative'] = '{0}'.format(TN + FP)
        record['Actuals Positive'] = '{0}'.format(FN + TP)

        record['Predicted Negative'] = '{0}'.format(TN + FN)
        record['Predicted Positive'] = '{0}'.format(FP + TP)

        record['Corrects'] = '{0}'.format(TN + TP)
        record['Incorrects'] = '{0}'.format(FN + FP)

        TNR = TN / (TN + FP)
        FPR = FP / (FP + TN)

        TPR = TP / (TP + FN)
        FNR = FN / (FN + TP)

        NPV = TN / (FN + TN)
        FOR = FN / (FN + TN)

        PPV = TP / (TP + FP)
        FDR = FP / (FP + TP)

        ACC = (TN + TP) / (TN + TP + FN + FP)
        F1 = 2 * PPV * TPR / (PPV + TPR)
        PRE = (TP + FN) / (TN + TP + FN + FP)

        record['TNR'] = '{0:.2%}'.format(TNR)
        record['FPR'] = '{0:.2%}'.format(FPR)

        record['TPR'] = '{0:.2%}'.format(TPR)
        record['FNR'] = '{0:.2%}'.format(FNR)

        record['NPV'] = '{0:.2%}'.format(NPV)
        record['FOR'] = '{0:.2%}'.format(FOR)

        record['PPV'] = '{0:.2%}'.format(PPV)
        record['FDR'] = '{0:.2%}'.format(FDR)

        record['ACC'] = '{0:.2%}'.format(ACC)
        record['F1'] = '{0:.2%}'.format(F1)
        record['PRE'] = '{0:.2%}'.format(PRE)

        record['ROC AUC Score'] = '{0:.2%}'.format(
            skm.roc_auc_score(
                y_true=y_true[:,i], 
                y_score=y_score[:,i],
                average='weighted'))

        record['Brier Score'] = '{0:.2%}'.format(
            skm.brier_score_loss(
                y_true=y_true[:,i], 
                y_prob=y_score[:,i]))

        record['Cross-Entropy / Log-Loss'] = '{0:.2%}'.format(
            skm.log_loss(
                y_true=y_true[:,i], 
                y_pred=y_score[:,i]))

        records.append(record)

    df_scores = pd.DataFrame.from_records(records)
    df_scores = df_scores.set_index('Class')
    
    return df_scores.transpose()

def confusion_matrix_values(threshold, y_true, y_score):
    # Returns TN, FP, FN, TP
    return skm.confusion_matrix(
        y_true=y_true, 
        y_pred=np.array(
            (np.array(y_score) > threshold),
            dtype=float),
        labels=[0.0, 1.0]).ravel()

def confusion_matrix_string(threshold, y_true, y_score):
    # Return the confusion matrix as a string
    tn, fp, fn, tp = confusion_matrix_values(
        threshold=threshold,
        y_true=y_true,
        y_score=y_score)
    
    return 'Threshold: {0:.4f}\n\nTN: {1}, FP: {2}\nFN: {3}, TP: {4}'.format(
        threshold,
        tn, 
        fp,
        fn,
        tp)

def metrics_string(threshold, y_true, y_score):
    
    # Return the confusion matrix as a string
    true_neg, false_pos, false_neg, true_pos = confusion_matrix_values(
        threshold=threshold,
        y_true=y_true,
        y_score=y_score)
    
    TN = float(true_neg)
    FP = float(false_pos)
    FN = float(false_neg)
    TP = float(true_pos)
    
    ACC = (TN + TP) / (TN + TP + FN + FP)
    PRE = (TP + FN) / (TN + TP + FN + FP)
    REC = score_f1(
        threshold=threshold,
        y_true=y_true,
        y_score=y_score)
    ROC = skm.roc_auc_score(y_true, y_score)
    BRI = skm.brier_score_loss(y_true, y_score)
    CRO = skm.log_loss(y_true, y_score)
    
    metrics = [
        ('Accuracy', ACC),
        ('Recall', REC),
        ('Prevalence', PRE),
        ('ROC AUC Score', ROC),
        ('Brier Score', BRI),
        ('Cross-Entropy / Log-Loss', CRO)
    ]
    
    message = ''
    
    for name, value in metrics:
        message += '{0}: {1:.4f}\n'.format(name, value)
        
    return message[:-1]

def calculate_tpr_fpr_prec(threshold, y_true, y_score):
    
    true_neg, false_pos, false_neg, true_pos = confusion_matrix_values(
        threshold=threshold,
        y_true=y_true,
        y_score=y_score)
    
    tpr_recall = float(true_pos) / (true_pos + false_neg)
    fpr = float(false_pos) / (false_pos + true_neg)
    precision = float(true_pos) / (true_pos + false_pos)
    
    return tpr_recall, fpr, precision

def score_f1(threshold, y_true, y_score):
    _, fp, fn, tp = confusion_matrix_values(
        threshold=threshold,
        y_true=y_true, 
        y_score=y_score)

    den = float(tp) + .5*(float(fp + fn))

    if den == 0.0:
        if float(tp) > 0.0:
            return 1.0
        else :
            return 0.0
    else:
        return (float(tp) / den)

def score_f1_invert(threshold, y_true, y_score):
    return 1 - score_f1(threshold, y_true, y_score)

def get_optimized_thresholds(y_true, y_score, score):
    # Get the threshold value minimizing the score function for each class
    thresholds = []
    
    for i in range(y_true.shape[1]):
        
        y_t = y_true[:, i] 
        y_s = y_score[:, i]

        result = sco.minimize_scalar(
            fun=score, 
            bounds=(0, 1), 
            args=(y_t, y_s),
            method='bounded',
            options={
                'xatol': 1e-5, 
                'maxiter': 250 })

        thresholds.append(
            result.x)
            
    return np.array(thresholds)

def predict_multiclass(model, thresholds, X):
    """Predict class labels for samples in X, by using the
    `predict_proba` model function and the optimized `thresholds`
    values (class specific)

    Args:
        model (estimator): scikit-learn style estimator implementing `predict_proba`.
        thresholds (ndarray, shape (nb_classes, )): list of optimized threshold values.
        X (ndarray, shape (n_samples, n_features)): Input data for prediction.

    Returns:
        [ndarray, shape [n_samples]]: Predicted class label per sample.
    """
    return predict_using_optimized_thresholds(
        thresholds=thresholds,
        y_score=model.predict_proba(X))

def predict_using_optimized_thresholds(thresholds, y_score):
    """[summary]

    Args:
        thresholds ([type]): [description]
        y_score ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Find the winners (scores higher than class-threshold).
    y_pred = np.array(
        (np.array(y_score) > thresholds),
        dtype=float)

    # Number of winners for each sample
    y_n_winners = y_pred.sum(
        axis=1,
        dtype=float)
    
    # Ratio of the scores compared to each class-threshold (how far are we from the threshold)
    y_t_ratio = np.array(
        y_score / thresholds,
        dtype=float)
    
    # Samples with no clear winner: proba ratio considering all classes (none beyond threshold)
    y_0w_vect = (1 - y_pred) * y_score * y_t_ratio

    # Samples with no clear winner: the class with the highest % wins
    y_0w_winner = np.array(
        (y_0w_vect == y_0w_vect.max(axis=1).reshape(-1, 1)),
        dtype=float)
    
    #Samples with multiple winners: proba ratio considering winning classes only (above threshold)
    y_nw_vect = y_pred * y_score * y_t_ratio
    
    # Samples with multiple winners: the class with the highest % wins
    y_nw_winner = np.array(
        (y_nw_vect == y_nw_vect.max(axis=1).reshape(-1, 1)),
        dtype=float)
    
    return np.where(
        (y_n_winners.reshape(-1, 1) == 1),
        y_pred, # The easy ones having 1 true winner
        np.where(
            (y_n_winners.reshape(-1, 1) == 0),
            y_0w_winner, # There is no winner
            y_nw_winner)) # There are multiple winners

def y_to_y_true(y):
    y_true = np.zeros(
        shape=(y.shape[0], y.max() + 1),
        dtype=float)

    for i in range(y.shape[0]):
        y_true[i, y[i]] = 1.0

    return y_true