# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
from scipy.stats import chi2_contingency

from .data import FairnessColumn, Dataset
from .reporter import BiasDetectionReporter


class FairnessMetric:
    fairness_col: FairnessColumn
    metric_name: str
    metric_ratio: float
    metric_ratio_str: str
    test_stat: float
    pvalue: float
    confusion_matrices: list

    def __init__(self, fairness_col: FairnessColumn, metric_name: str,
                 metric_ratio_str: str):
        self.fairness_col = fairness_col
        self.metric_name = metric_name
        self.metric_ratio_str = metric_ratio_str


class BiasDetector:
    def __init__(self, ds: Dataset):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DESCRIPTION.
        fairness_cols : list
            DESCRIPTION.
        outcome_col : str
            Predicted label.
        label_col : str, optional
            True label. The default is None.

        Returns
        -------
        None.

        """
        self.ds = ds
        self.df = ds.df
        self.fairness_cols = ds.fairness_cols
        self.outcome_col = ds.outcome_col
        self.label_col = ds.label_col

    def get_confusion_matrices(self, col: FairnessColumn) -> list:
        """
        Parameters
        ----------
        col : FairnessColumn
            DESCRIPTION.

        Returns
        -------
        ret : list of two confusion matrices for the protected and the
            reference column. Each confusion matrix is a numpy array
            of the form [[tp, fn], [fp, tn]]
        """
        ret = []
        df = self.df
        for col_name in [col.prot_col, col.ref_col]:
            tp = len(df[(df[self.label_col] == 1) &
                        (df[self.outcome_col] == 1) &
                        (df[col_name] == 1)])
            fn = len(df[(df[self.label_col] == 1) &
                        (df[self.outcome_col] == 0) &
                        (df[col_name] == 1)])
            fp = len(df[(df[self.label_col] == 0) &
                        (df[self.outcome_col] == 1) &
                        (df[col_name] == 1)])
            tn = len(df[(df[self.label_col] == 0) &
                        (df[self.outcome_col] == 0) &
                        (df[col_name] == 1)])
            ret.append(np.array([[tp, fn], [fp, tn]]))

        return ret

    def get_equalized_odds(self, col: FairnessColumn):
        ret = FairnessMetric(fairness_col=col, metric_name='Equalized Odds',
                             metric_ratio_str='naTP(A=1)/naTP(A=0)'
                             '+ naFP(A=1)/naFP(A=0)')
        cf_prot, cf_ref = self.get_confusion_matrices(col)

        # Normalize cf_prot and cf_ref for each actual value
        #   (note that each matrix is in the form [[tp, fn], [fp, tn]]
        cf_prot_na = cf_prot / np.linalg.norm(cf_prot, ord=1, axis=1,
                                              keepdims=True)
        cf_ref_na = cf_ref / np.linalg.norm(cf_ref, ord=1, axis=1,
                                            keepdims=True)
        ret.metric_ratio = ((cf_prot_na[0, 0]/cf_ref_na[0, 0]) +
                            (cf_prot_na[1, 0]/cf_ref_na[1, 0]))
        ret.confusion_matrices = [cf_prot_na, cf_ref_na]

        # The contingency table should be organized with the rows representing
        #   one categorical variable and the columns representing the other
        #   categorical variable.
        cont_1 = np.zeros((2, 2))
        cont_1[0, :] = cf_prot[0, :]
        cont_1[1, :] = cf_ref[0, :]
        stat_1, pvalue_1, dof_1, expected_1 = chi2_contingency(cont_1)

        cont_2 = np.zeros((2, 2))
        cont_2[0, :] = cf_prot[1, :]
        cont_2[1, :] = cf_ref[1, :]
        stat_2, pvalue_2, dof_2, expected_2 = chi2_contingency(cont_2)

        ret.pvalue = pvalue_1 * pvalue_2
        ret.test_stat = (stat_1, stat_2)
        return ret

    def get_equal_opportunity(self, col: FairnessColumn):
        ret = FairnessMetric(fairness_col=col,
                             metric_name='Equal Opportunity',
                             metric_ratio_str='naTP(A=1)/naTP(A=0)')
        cf_prot, cf_ref = self.get_confusion_matrices(col)

        # Normalize cf_prot and cf_ref for each actual value
        #   (note that each matrix is in the form [[tp, fn], [fp, tn]]
        cf_prot_na = cf_prot / np.linalg.norm(cf_prot, ord=1, axis=1,
                                              keepdims=True)
        cf_ref_na = cf_ref / np.linalg.norm(cf_ref, ord=1, axis=1,
                                            keepdims=True)
        ret.metric_ratio = cf_prot_na[0, 0]/cf_ref_na[0, 0]
        ret.confusion_matrices = [cf_prot_na, cf_ref_na]

        # The contingency table should be organized with the rows representing
        #   one categorical variable and the columns representing the other
        #   categorical variable.
        cont_1 = np.zeros((2, 2))
        cont_1[0, :] = cf_prot[0, :]
        cont_1[1, :] = cf_ref[0, :]
        stat_1, pvalue_1, dof_1, expected_1 = chi2_contingency(cont_1)

        ret.pvalue = pvalue_1
        ret.test_stat = (stat_1,)
        return ret

    def get_demographic_parity(self, col: FairnessColumn):
        ret = FairnessMetric(fairness_col=col,
                             metric_name='Demographic Parity',
                             metric_ratio_str='nP(A=1)/nP(A=0)')
        cf_prot, cf_ref = self.get_confusion_matrices(col)

        marginal_prot = cf_prot.sum(axis=0)
        marginal_prot_n = marginal_prot / marginal_prot.sum()
        marginal_ref = cf_ref.sum(axis=0)
        marginal_ref_n = marginal_prot / marginal_prot.sum()
        ret.metric_ratio = marginal_prot_n[0]/marginal_ref_n[0]
        ret.confusion_matrices = [marginal_prot_n, marginal_ref_n]

        # The contingency table should be organized with the rows representing
        #   one categorical variable and the columns representing the other
        #   categorical variable.
        cont_1 = np.zeros((2, 2))
        cont_1[0, :] = marginal_prot
        cont_1[1, :] = marginal_ref
        stat_1, pvalue_1, dof_1, expected_1 = chi2_contingency(cont_1)

        ret.pvalue = pvalue_1
        ret.test_stat = (stat_1,)
        return ret

    def get_calibration_fairness(self, col: FairnessColumn):
        ret = FairnessMetric(fairness_col=col,
                             metric_name='Calibration Fairness',
                             metric_ratio_str='npTP(A=1)/npTP(A=0)'
                             ' + npFN(A=1)/npFN(A=0)')
        cf_prot, cf_ref = self.get_confusion_matrices(col)

        # Normalize cf_prot and cf_ref for each predicted value
        #   (note that each matrix is in the form [[tp, fn], [fp, tn]]
        cf_prot_np = cf_prot / np.linalg.norm(cf_prot, ord=1, axis=0,
                                              keepdims=True)
        cf_ref_np = cf_ref / np.linalg.norm(cf_ref, ord=1, axis=0,
                                            keepdims=True)
        ret.metric_ratio = ((cf_prot_np[0, 0]/cf_ref_np[0, 0]) +
                            (cf_prot_np[1, 0]/cf_ref_np[1, 0]))
        ret.confusion_matrices = [cf_prot_np, cf_ref_np]

        # The contingency table should be organized with the rows representing
        #   one categorical variable and the columns representing the other
        #   categorical variable.
        cont_1 = np.zeros((2, 2))
        cont_1[:, 0] = cf_prot[:, 0]
        cont_1[:, 1] = cf_ref[:, 0]
        stat_1, pvalue_1, dof_1, expected_1 = chi2_contingency(cont_1)

        cont_2 = np.zeros((2, 2))
        cont_2[:, 0] = cf_prot[:, 1]
        cont_2[:, 1] = cf_ref[:, 1]
        stat_2, pvalue_2, dof_2, expected_2 = chi2_contingency(cont_2)

        ret.pvalue = pvalue_1 * pvalue_2
        ret.test_stat = (stat_1, stat_2)
        return ret

    def get_differential_validity(self, col: FairnessColumn):
        ret = FairnessMetric(fairness_col=col,
                             metric_name='Differential Validity',
                             metric_ratio_str='ndTP(A=1)/ndTP(A=0)')
        cf_prot, cf_ref = self.get_confusion_matrices(col)

        diag_prot = cf_prot.diagonal()
        diag_prot_n = diag_prot / diag_prot.sum()
        diag_ref = cf_ref.diagonal()
        diag_ref_n = diag_ref / diag_ref.sum()
        ret.metric_ratio = diag_prot[0]/diag_ref[0]
        ret.confusion_matrices = [diag_prot_n, diag_ref_n]

        # The contingency table should be organized with the rows representing
        #   one categorical variable and the columns representing the other
        #   categorical variable.
        cont_1 = np.zeros((2, 2))
        cont_1[0, :] = diag_prot
        cont_1[:, 1] = diag_ref
        stat_1, pvalue_1, dof_1, expected_1 = chi2_contingency(cont_1)

        ret.pvalue = pvalue_1
        ret.test_stat = (stat_1,)
        return ret

    def get_metrics(self, metric_methods='all'):
        all_metric_methods = ['get_equalized_odds',
                              'get_equal_opportunity',
                              'get_demographic_parity',
                              'get_calibration_fairness',
                              'get_differential_validity']

        assert (metric_methods == 'all'
                or set(metric_methods).issubset(all_metric_methods))
        if metric_methods == 'all':
            metric_methods = all_metric_methods
        ret = defaultdict(dict)
        for col in self.fairness_cols:
            for metric_method in metric_methods:
                ret[col][metric_method] = getattr(self, metric_method)(col)

        return ret

    def generate_report(self, metric_methods='all', show_graphs=True,
                        output_format='raw'):
        assert output_format in ['raw']
        res = self.get_metrics(metric_methods)
        reporter = BiasDetectionReporter(self.ds, res)
        if output_format == 'raw':
            reporter.raw_report()
