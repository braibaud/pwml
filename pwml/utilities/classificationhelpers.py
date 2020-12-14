import math as m
import pandas as pd
import numpy as np

import matplotlib as mat
from matplotlib import pyplot as plt
import pylab as pyl
import seaborn as sns

from scipy import optimize as sco

import sklearn as sk
from sklearn import exceptions as skx
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
from sklearn.utils import class_weight as skcw

from . import commonhelpers as cmn


# Class holding static properties only
class GraphicsStatics(object):
    
    # Globals
    g_palette = None
    
    g_landscape_fig_size = (20, 10)
    g_square_fig_size = (20, 20)
    g_portrait_fig_size = (20, 30)

    g_styles_initialized = False

    @staticmethod
    def initialize_matplotlib_styles():

        if not GraphicsStatics.g_styles_initialized:
            sns.set()
            sns.set_style('whitegrid', {'axes.facecolor': '.9'})
            plt.style.use('fivethirtyeight')
            sns.set_context('talk')

            # Assign the default palette
            GraphicsStatics.g_palette = sns.color_palette('Set2', 8)

            sns.set_palette(
                GraphicsStatics.g_palette)

            pyl.rcParams['figure.figsize'] = GraphicsStatics.g_landscape_fig_size
            plt.rcParams['figure.figsize'] = GraphicsStatics.g_landscape_fig_size
            plt.rcParams['axes.labelsize'] = 18
            plt.rcParams['axes.titlesize'] = 18

            GraphicsStatics.g_styles_initialized = True


class MulticlassClassifierOptimizer(object):

    def __init__(self, model, classes, scoring_function):

        if not MulticlassClassifierOptimizer.optimizable_model(model):
            raise NotImplementedError('The model does not implement method "predict_proba" and cannot be optimized.')

        self.model = model
        self.classes = classes
        self.scoring_function = scoring_function

        self.optimized = False
        self.thresholds = None

    @property
    def estimator_(self):

        self.assert_optimized()

        return MulticlassClassifierOptimizer.get_estimator(
            model=self.model)

    @property
    def classes_(self):
        return self.estimator_.classes_

    @property
    def coef_(self):
        return self.estimator_.coef_

    @property
    def intercept_(self):
        return self.estimator_.intercept_

    def fit(self, X, y):

        if not MulticlassClassifierOptimizer.fitted_model(self.model):

            print('      -> Fitting base model (wasn\'t fitted).')

            self.model.fit(
                X=X,
                y=y)

        print('      -> Model calibration.')

        self.model = skc.CalibratedClassifierCV(
            base_estimator=self.model,
            method='sigmoid',
            cv='prefit')

        self.model.fit(
            X=X,
            y=y,
            sample_weight=skcw.compute_sample_weight(
                class_weight='balanced', 
                y=y))

        print('      -> Optimizing multiclass thresholds.')

        self.thresholds = MulticlassClassifierOptimizer.get_optimized_thresholds(
            scoring_function=self.scoring_function,
            y_true=MulticlassClassifierOptimizer.one_hot_encode(
                y=y),
            y_score=self.model.predict_proba(
                X=X))

        self.optimized = True

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X (ndarray, shape (n_samples, n_features)): The input samples.
        
        Returns:
            ndarray, shape (n_samples, n_classes): The class probabilities 
            of the input samples. The order of the classes corresponds to that
            in the attribute :term:`classes_`.
        """
        self.assert_optimized()

        return self.model.predict_proba(X)

    def predict(self, X):
        """Predict class labels for samples in X, by using the
        `predict_proba` model function and the optimized `thresholds`
        values (class specific)

        Args:
            X (ndarray, shape (n_samples, n_features)): Input data for prediction.

        Returns:
            ndarray, shape (n_samples, n_classes): Predicted class label per sample (1-hot encoded).
        """
        self.assert_optimized()

        return self.predict_from_score(
            thresholds=self.thresholds,
            y_score=self.model.predict_proba(X))

    def score(self, X, y):

        self.assert_optimized()

        class_weights = 1/skcw.compute_class_weight(
            class_weight='balanced', 
            classes=np.arange(
                start=0,
                stop=len(self.classes),
                step=1,
                dtype=int), 
            y=y)

        y_true = MulticlassClassifierOptimizer.one_hot_encode(y=y)
        y_score = self.predict_proba(X)

        scores = []

        for index, _ in enumerate(self.classes):

            scores.append(
                self.scoring_function(
                    threshold=self.thresholds[index],
                    y_true=y_true[:, index],
                    y_score=y_score[:, index]))

        return np.average(
            a=scores,
            weights=class_weights)

    def assert_optimized(self):
        if not self.optimized:
            raise NotImplementedError('The model has not been optimized yet.')

    def predict_from_score_1c(self, y_score, class_index, class_threshold):
        updated_thresholds = self.thresholds.copy()
        updated_thresholds[class_index] = class_threshold

        y_pred = MulticlassClassifierOptimizer.predict_from_score(
            thresholds=updated_thresholds,
            y_score=y_score)

        return y_pred[:, class_index]

    def confusion_matrix_1c(self, y_true, y_score, class_index, class_threshold):
        return skm.confusion_matrix(
            y_true=y_true[:, class_index], 
            y_pred=self.predict_from_score_1c(
                y_score=y_score, 
                class_index=class_index, 
                class_threshold=class_threshold),
            labels=[0.0, 1.0]).ravel()

    def sort_classes_by_score(self, X, y, n_top=10):

        y_true = MulticlassClassifierOptimizer.one_hot_encode(y)
        y_score = self.predict_proba(X)

        records = []

        for index, name in enumerate(self.classes):
            records.append({
                'Class': name,
                'Index': index,
                'Score': self.scoring_function(
                    self.thresholds[index],
                    y_true[:, index], 
                    y_score[:, index])
            })

        df_scores = pd.DataFrame.from_records(records)
        df_scores = df_scores.convert_dtypes()
        df_scores = df_scores.sort_values(by='Score', ascending=True)

        return list(df_scores.head(n_top)['Class'].values)

    def get_metrics_by_class(self, X, y, transpose=False):
        return self.get_metrics_by_class_base(
            y_true=MulticlassClassifierOptimizer.one_hot_encode(y),
            y_score=self.predict_proba(X),
            transpose=transpose)

    def get_metrics_by_class_base(self, y_true, y_score, transpose=False):
        
        records = []

        for index, name in enumerate(self.classes):

            tn, fp, fn, tp = self.confusion_matrix_1c(
                y_true=y_true,
                y_score=y_score,
                class_index=index,
                class_threshold=self.thresholds[index])

            record = {
                'Class': name,
                'True Negative': tn,
                'False Positive': fp,
                'False Negative': fn,
                'True Positive': tp,
                'Prevalence': BinaryClassifierHelper.prevalence(tn, fp, fn, tp),
                'Actual Negative': tn + fp,
                'Actual Positive': fn + tp,
                'Predicted Negative': tn + fn,
                'Predicted Positive': fp + tp,
                'Predicted Correctly': tn + tp,
                'Predicted Incorrectly': fn + fp,
                'Accuracy': float(tn + tp) / float(tn + fp + fn + tp),
                'Precision': BinaryClassifierHelper.precision(tn, fp, fn, tp),
                'Recall': BinaryClassifierHelper.recall(tn, fp, fn, tp),
                'F1-Score': BinaryClassifierHelper.f1(tn, fp, fn, tp),
                'Fallout': BinaryClassifierHelper.fallout(tn, fp, fn, tp),
                'ROC AUC Score': skm.roc_auc_score(y_true[:, index], y_score[:, index]),
                'Brier Score': skm.brier_score_loss(y_true[:, index], y_score[:, index]),
                'Cross-Entropy': skm.log_loss(y_true[:, index], y_score[:, index]),
            }

            records.append(record)

        df_scores = pd.DataFrame.from_records(records)
        df_scores = df_scores.set_index('Class')
        df_scores = df_scores.convert_dtypes()

        if transpose:
            return df_scores.transpose()
        else:
            return df_scores

    def plot_curve(self, class_name, X, y, n_bins=10):

        if class_name in self.classes:

            y_true = MulticlassClassifierOptimizer.one_hot_encode(y)
            y_score = self.predict_proba(X)

            index = self.classes.index(class_name)

            BinaryClassifierHelper.plot_curves(
                title='{0}\nScore: {1:.4f}'.format(
                    class_name.upper(), 
                    self.scoring_function(
                        threshold=self.thresholds[index],
                        y_true=y_true[:, index],
                        y_score=y_score[:, index])),
                y_true=y_true[:, index], 
                y_score=y_score[:, index],
                best_threshold=self.thresholds[index],
                n_bins=n_bins)

    def plot_curves(self, X, y, n_bins=10, n_top=10):

        classes = None

        if n_top is not None:
            classes = self.sort_classes_by_score(
                X=X,
                y=y,
                n_top=n_top)

        self.plot_curves_base(
            y_true=MulticlassClassifierOptimizer.one_hot_encode(y),
            y_score=self.predict_proba(X),
            n_bins=n_bins,
            classes=classes)

    def plot_curves_base(self, y_true, y_score, n_bins=10, classes=None):
        
        for index, name in enumerate(self.classes):
            if (classes is None or len(classes) == 0) or (classes is not None and name in classes):
                BinaryClassifierHelper.plot_curves(
                    title='{0}\nScore: {1:.4f}'.format(
                        name.upper(), 
                        self.scoring_function(
                            threshold=self.thresholds[index],
                            y_true=y_true[:, index],
                            y_score=y_score[:, index])),
                    y_true=y_true[:, index], 
                    y_score=y_score[:, index],
                    best_threshold=self.thresholds[index],
                    n_bins=n_bins)

    @staticmethod
    def optimizable_model(model):
        return callable(getattr(model, 'predict_proba', None))

    @staticmethod
    def predict_from_score(thresholds, y_score):
        """Predict class labels for scores and the optimized `thresholds`
        values (class specific)

        Args:
            y_score (ndarray, shape (n_samples, n_classes)): Input data for prediction.

        Returns:
            ndarray, shape (n_samples, n_classes): Predicted class label per sample (1-hot encoded).
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

    @staticmethod
    def one_hot_encode(y):
        y_true = np.zeros(
            shape=(y.shape[0], y.max() + 1),
            dtype=float)

        for i in range(y.shape[0]):
            y_true[i, y[i]] = 1.0

        return y_true

    @staticmethod
    def get_optimized_thresholds(scoring_function, y_true, y_score):
        
        # The objective function is the 
        objective = lambda threshold, y_true, y_score: 1 - scoring_function(threshold, y_true, y_score)

        # Define alpha
        alpha = 0.05

        # Get the threshold value minimizing the score function for each class
        thresholds = []
        
        for i in range(y_true.shape[1]):
            
            result = sco.minimize_scalar(
                fun=objective, 
                bounds=(0 + alpha, 1 - alpha), 
                args=(y_true[:, i], y_score[:, i]),
                method='bounded',
                options={
                    'xatol': 1e-5, 
                    'maxiter': 250 })

            thresholds.append(
                result.x)
                
        return np.array(thresholds)

    @staticmethod
    def get_estimator(model):
        
        estimator = model

        if type(model).__name__ == 'Pipeline':
            _, estimator = model.steps[-1]

        return estimator

    @staticmethod
    def fitted_model(model):
        
        estimator = MulticlassClassifierOptimizer.get_estimator(
            model=model)
        
        try:
            skuv.check_is_fitted(estimator)
            return True
        except skx.NotFittedError:
            return False


class BinaryClassifierHelper(object):

    @staticmethod
    def confusion_matrix(threshold, y_true, y_score):
        return skm.confusion_matrix(
            y_true=y_true, 
            y_pred=np.array(
                (np.array(y_score) > threshold),
                dtype=float),
            labels=[0.0, 1.0]).ravel()

    @staticmethod
    def f1_score(threshold, y_true, y_score):
        return BinaryClassifierHelper.f1(
            *BinaryClassifierHelper.confusion_matrix(
                threshold=threshold,
                y_true=y_true, 
                y_score=y_score))

    @staticmethod
    def get_confusion_matrix_string(threshold, y_true, y_score):
        tn, fp, fn, tp = BinaryClassifierHelper.confusion_matrix(
            threshold=threshold,
            y_true=y_true,
            y_score=y_score)

        metrics = [
            ('Threshold', threshold, '.4f'),
            ('TN', tn, 'd'),
            ('FP', fp, 'd'),
            ('FN', fn, 'd'),
            ('TP', tp, 'd')
        ]
            
        return cmn.flatten_namevalue_pairs(
            pairs=metrics,
            separator='\n')

    @staticmethod
    def get_metrics_string(threshold, y_true, y_score):
        tn, fp, fn, tp = BinaryClassifierHelper.confusion_matrix(
            threshold=threshold,
            y_true=y_true,
            y_score=y_score)
        
        metrics = [
            ('Accuracy', BinaryClassifierHelper.accuracy(tn, fp, fn, tp), '.4f'),
            ('Recall', BinaryClassifierHelper.recall(tn, fp, fn, tp), '.4f'),
            ('Precision', BinaryClassifierHelper.precision(tn, fp, fn, tp), '.4f'),
            ('Fallout', BinaryClassifierHelper.fallout(tn, fp, fn, tp), '.4f'),
            ('Prevalence', BinaryClassifierHelper.prevalence(tn, fp, fn, tp), '.4f'),
            ('ROC AUC Score', skm.roc_auc_score(y_true, y_score), '.4f'),
            ('Brier Score', skm.brier_score_loss(y_true, y_score), '.4f'),
            ('Cross-Entropy', skm.log_loss(y_true, y_score), '.4f')
        ]
            
        return cmn.flatten_namevalue_pairs(
            pairs=metrics,
            separator='\n')

    @staticmethod
    def recall(tn, fp, fn, tp):
        if tp + fn == 0:
            return float(1)
        else:
            return float(tp) / float(tp + fn)

    @staticmethod
    def fallout(tn, fp, fn, tp):
        if tn == 0:
            if fp > 0:
                return float(1)
            else:
                return float(0)
        else:
            return float(fp) / float(fp + tn)

    @staticmethod
    def precision(tn, fp, fn, tp):
        if tp == 0:
            if fp > 0:
                return float(0)
            else:
                return float(1)
        else:
            return float(tp) / float(tp + fp)

    @staticmethod
    def f1(tn, fp, fn, tp):
        den = float(tp) + .5*(float(fp + fn))

        if den == 0.0:
            return float(1)
        else:
            return float(tp) / den

    @staticmethod
    def accuracy(tn, fp, fn, tp):
        if tn + fp + fn + tp == 0:
            return float(0)
        else:
            return float(tp + tn) / float(tn + fp + fn + tp)

    @staticmethod
    def prevalence(tn, fp, fn, tp):
        if tn + fp + fn + tp == 0:
            return float(0)
        else:
            return float(tp + fn) / float(tn + fp + fn + tp)

    @staticmethod
    def calculate_tpr_fpr_prec(threshold, y_true, y_score):
        tn, fp, fn, tp = BinaryClassifierHelper.confusion_matrix(
            threshold=threshold,
            y_true=y_true,
            y_score=y_score)
        
        recall_ = BinaryClassifierHelper.recall(tn, fp, fn, tp)
        fallout_ = BinaryClassifierHelper.fallout(tn, fp, fn, tp)
        precision_ = BinaryClassifierHelper.precision(tn, fp, fn, tp)
        
        return recall_, fallout_, precision_

    @staticmethod
    def plot_curves(title, y_true, y_score, best_threshold=None, n_bins=10):

        # syle
        GraphicsStatics.initialize_matplotlib_styles()

        roc_color = 'crimson'
        pr_color = 'royalblue'
        main_linestyle = 'solid'
        neutral_color = 'k'
        neutral_linestyle = 'dashed'
        lw = 2

        fig = plt.figure(
            figsize=GraphicsStatics.g_portrait_fig_size)

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
        
        ax[0].set_xlabel('Fallout', fontweight='bold')
        
        ax[0].set_ylabel('Recall', fontweight='bold')
        
        ax[0].set_title(
            'AUROC = {0:.2f}'.format(
                skm.auc(
                    x=fpr, 
                    y=tpr_recall)),
            fontweight='bold')

        #Plot dots at certain decision thresholds, for clarity
        for threshold in [.1, .3, .5, .7, .9]:
            
            tpr_recall, fpr, _ = BinaryClassifierHelper.calculate_tpr_fpr_prec(
                threshold=threshold,
                y_true=y_true,
                y_score=y_score)
            
            ax[0].plot(
                fpr, 
                tpr_recall, 
                'o', 
                color=roc_color)
            
            ax[0].annotate(
                '  t = {0:.1f}  '.format(threshold), 
                (fpr, tpr_recall),
                ha='right',
                rotation=-45,
                size=16)

            
        precision, tpr_recall, _ = skm.precision_recall_curve(
            y_true=y_true,
            probas_pred=y_score)
            
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
            
        ax[1].set_xlabel('Recall', fontweight='bold')
            
        ax[1].set_ylabel('Precision', fontweight='bold')
            
        ax[1].set_xlim([-0.1, 1.1])
            
        ax[1].set_ylim([-0.1, 1.1])
            
        ax[1].set_title(
            'Average Precision = {0:.2f}'.format(
                skm.average_precision_score(
                    y_true=y_true,
                    y_score=y_score)), 
            fontweight='bold')

        #Plot dots at certain decision thresholds, for clarity
        for threshold in [.1, .3, .5, .7, .9]:
            
            tpr_recall, _, precision = BinaryClassifierHelper.calculate_tpr_fpr_prec(
                threshold=threshold,
                y_true=y_true,
                y_score=y_score)
            
            ax[1].plot(
                tpr_recall, 
                precision, 
                'o',
                color=pr_color)
            
            ax[1].annotate(
                '  t = {0:.1f}  '.format(threshold), 
                (tpr_recall, precision),
                ha='left',
                rotation=45,
                size=16)

        # Recall / Precision: xing curves
        threshold_value = list(np.linspace(
            start=0.0, 
            stop=1.0, 
            num=200))
        
        tpr_v = []
        ppv_v = []
        
        for threshold in threshold_value:
            
            tpr, _, ppv = BinaryClassifierHelper.calculate_tpr_fpr_prec(
                threshold=threshold,
                y_true=y_true,
                y_score=y_score)
            
            tpr_v.append(tpr)
            ppv_v.append(ppv)
                    
        ax[2].plot(
            threshold_value,
            tpr_v,
            label='Recall')

        ax[2].plot(
            threshold_value, 
            ppv_v, 
            label='Precision')

        if best_threshold is not None:
            
            x_opt = best_threshold

            tpr_opt, _, ppv_opt = BinaryClassifierHelper.calculate_tpr_fpr_prec(
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

        fig.text(
            x=0.02, 
            y=1, 
            s=BinaryClassifierHelper.get_confusion_matrix_string(
                threshold=best_threshold,
                y_true=y_true,
                y_score=y_score), 
            ha='left', 
            va='top',
            fontsize=14,
            fontstyle='italic',
            bbox={ 
                'facecolor': GraphicsStatics.g_palette[5], 
                'alpha': 0.5, 
                'pad': 7
            })

        fig.text(
            x=.95, 
            y=1, 
            s=BinaryClassifierHelper.get_metrics_string(
                threshold=best_threshold,
                y_true=y_true,
                y_score=y_score), 
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
