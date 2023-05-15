# -*- coding: utf-8 -*-
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import numpy.random

# from .detector import FairnessMetric
from .data import Dataset

class BiasDetectionReporter:
    def __init__(self, ds: Dataset, detection_results: dict,
                 pvalue_tr: float = .05):
        self.ds = ds
        self.detection_results = detection_results
        self.pvalue_tr = pvalue_tr

    def raw_report(self, show_graphs=True):
        ds_name = self.ds.dataset_name
        if ds_name == '':
            ds_name = '_unnamed_'
        print('*** Full bias detection report for [%s] dataset ***' % ds_name)
        print(datetime.now().strftime("%Y-%m-%d %H:%M"))
        print('*legend for method ratios: ???\n')
        for col, method_results in self.detection_results.items():
            print('Pairwise bias detection results for %s vs %s in group %s' %
                  (col.prot_col, col.ref_col, col.group))
            get_equalized_odds = False
            get_equal_opportunity = False
            for method, res in method_results.items():
                print('\t%s%s: %s=%.2f, pvalue=%.2f' %
                      (res.metric_name,
                       '*' if res.pvalue < self.pvalue_tr else '',
                       res.metric_ratio_str, res.metric_ratio, res.pvalue))
                if show_graphs:
                    if method in ['get_equalized_odds',
                                  'get_equal_opportunity',
                                  'get_demographic_parity']:
                        if method == 'get_equalized_odds':
                            if get_equal_opportunity:
                                print('\tSee the plot of Equal Opportunity.\n')
                                continue
                            else:
                                get_equalized_odds = True
                        elif method == 'get_equal_opportunity':
                            if get_equalized_odds:
                                print('\tSee the plot of Equalized Odds.\n')
                                continue
                            else:
                                get_equal_opportunity = True
                        self.vis_rowwise(res)
                    elif method == 'get_calibration_fairness':
                        self.vis_colwise(res)
                    elif method == 'get_differential_validity':
                        self.vis_diag(res)
                print()

    def vis_rowwise(self, fairness_metric):
        fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])

        for i, ax in enumerate(axn.flat):
            arr = fairness_metric.confusion_matrices[i]
            if len(arr.shape) == 2:
                arow = arr[0, :]
            else:
                arow = arr
            sn.heatmap(arow[np.newaxis, :], ax=ax, cbar=i == 0,
                       vmin=0, vmax=1, cbar_ax=None if i else cbar_ax,
                       annot=True, square=True, fmt='.2f')
            ax.invert_yaxis()
            ax.set_title(fairness_metric.fairness_col.prot_col if i == 0
                         else fairness_metric.fairness_col.ref_col)
            fig.text(0.5, .3, 'ŷ', ha='center')
            fig.text(-0.04, 0.5, 'y=1', va='center', rotation='vertical')
            t = '%s (%s)' % (fairness_metric.fairness_col.group,
                             fairness_metric.metric_name)
            fig.text(0.5, .8, t, ha='center')

        plt.show()

    def vis_colwise(self, fairness_metric):
        fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])

        for i, ax in enumerate(axn.flat):
            arr = fairness_metric.confusion_matrices[i]
            acol = arr[:, 0]
            sn.heatmap(acol[:, np.newaxis], ax=ax, cbar=i == 0,
                       vmin=0, vmax=1, cbar_ax=None if i else cbar_ax,
                       annot=True, square=True, fmt='.2f')
            ax.invert_yaxis()
            ax.set_title(fairness_metric.fairness_col.prot_col if i == 0
                         else fairness_metric.fairness_col.ref_col)
            fig.text(0.5, .2, 'ŷ=1', ha='center')
            fig.text(-0.04, 0.5, 'y', va='center', rotation='vertical')
            t = '%s (%s)' % (fairness_metric.fairness_col.group,
                             fairness_metric.metric_name)
            fig.text(0.5, .8, t, ha='center')

        plt.show()

    def vis_diag(self, fairness_metric):
        fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)
        cbar_ax = fig.add_axes([.91, .3, .03, .4])

        for i, ax in enumerate(axn.flat):
            arr = fairness_metric.confusion_matrices[i]
            adiag = [[arr[0], 1], [1, arr[1]]]
            adiag = np.around(adiag, 2)
            annot = [[str(adiag[0][0]), ''], ['', str(adiag[1][1])]]
            sn.heatmap(adiag, ax=ax, cbar=i == 0, vmin=0, vmax=1,
                       cbar_ax=None if i else cbar_ax, annot=annot,
                       square=True, fmt='')
            ax.invert_yaxis()
            ax.set_title(fairness_metric.fairness_col.prot_col if i == 0
                         else fairness_metric.fairness_col.ref_col)
            fig.text(0.5, .2, 'ŷ', ha='center')
            fig.text(-0.04, 0.5, 'y', va='center', rotation='vertical')
            t = '%s (%s)' % (fairness_metric.fairness_col.group,
                             fairness_metric.metric_name)
            fig.text(0.5, .8, t, ha='center')

        plt.show()


class BiasDetectionReporterOverTime:
    def __init__(self, ds: Dataset, detection_results: dict,
                 pvalue_tr: float = .05):
        self.ds = ds
        self.detection_results = detection_results
        self.pvalue_tr = pvalue_tr
