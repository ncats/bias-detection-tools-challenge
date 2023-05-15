import json
import logging
import argparse
from collections import defaultdict
from itertools import repeat, chain
from typing import Union

import numpy as np
import pandas as pd

from tabulate import tabulate
from sklearn.metrics import roc_curve, confusion_matrix
from fairlearn.metrics import (
    equalized_odds_difference,
    equalized_odds_ratio,
    demographic_parity_difference,
    demographic_parity_ratio
)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Enables serialization of numpy types
        :param obj:
        :return:
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self)


def measure(
        inp: Union[str, pd.DataFrame],
        binary_outcome_column: str,
        protected_classes: list[str],
        reference_classes: list[str] = None,
        probability_column: str = None,
        y_pred_column: str = None,
        sample_weights_column: str = None,
        pos_outcome_indicator: str = '1',
        debug: bool = False,
        extra_display_cols: list[tuple[str, str]] = None,
        suppress_output: bool = False
):
    input_df: pd.DataFrame = (pd.read_csv(inp, dtype={binary_outcome_column: str})
                              if isinstance(inp, str) else inp)
    if not pd.api.types.is_string_dtype(input_df[binary_outcome_column]):
        input_df[binary_outcome_column] = input_df[binary_outcome_column].astype(str)
    if not isinstance(pos_outcome_indicator, str):
        pos_outcome_indicator = str(pos_outcome_indicator)
    sample_weights = None if sample_weights_column is None else input_df[sample_weights_column]
    y_true = input_df[binary_outcome_column] == pos_outcome_indicator.strip()
    best_threshold = None
    if y_pred_column is None:
        y_prob = input_df[probability_column]

        fprs_from_roc, tprs_from_roc, thresholds = roc_curve(y_true, y_prob, sample_weight=sample_weights)
        distances = np.sqrt((1 - tprs_from_roc) ** 2 + fprs_from_roc ** 2)

        best_idx = np.argmin(distances)
        best_threshold = thresholds[best_idx]

        y_pred = input_df[probability_column] >= best_threshold
    else:
        y_pred = input_df[y_pred_column]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, sample_weight=sample_weights).ravel()
    sensitive_features = input_df[protected_classes]
    metrics = dict(
        num_samples=input_df.shape[0],
        false_positive_rate=fp / (fp + tn),
        false_negative_rate=fn / (fn + tp),
        true_positive_rate=tp / (tp + fn),
        true_negative_rate=tn / (tn + fp),
        all_protected_equal_odds_difference=equalized_odds_difference(
            y_true,
            y_pred,
            sensitive_features=sensitive_features,
            sample_weight=sample_weights
        ),
        all_protected_equal_odds_ratio=equalized_odds_ratio(
            y_true,
            y_pred,
            sensitive_features=sensitive_features,
            sample_weight=sample_weights
        ),
        all_protected_demographic_parity_difference=demographic_parity_difference(
            y_true,
            y_pred,
            sensitive_features=sensitive_features,
            sample_weight=sample_weights
        ),
        all_protected_demographic_parity_ratio=demographic_parity_ratio(
            y_true,
            y_pred,
            sensitive_features=sensitive_features,
            sample_weight=sample_weights,
        ),
        **{
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    )

    for class_type, class_ in chain(
            zip(repeat('protected'), protected_classes),
            zip(repeat('reference'), reference_classes)
    ):
        k_ = f"{class_type}_class_metrics"
        if k_ not in metrics:
            metrics[k_] = defaultdict(dict)
        protected_df = (input_df[input_df[[class_] + reference_classes].any(axis=1)]
                        if reference_classes else input_df)
        y_true_protected = (protected_df[binary_outcome_column] == pos_outcome_indicator.strip()
                            if reference_classes else y_true)
        if best_threshold is not None:
            y_pred_protected = ((protected_df[probability_column] >= best_threshold)
                                if reference_classes else y_pred)
        else:
            y_pred_protected = ((protected_df[y_pred_column])
                                if reference_classes else y_pred)
        protected_class_metrics = calculate_class_metrics(y_true_protected, y_pred_protected, protected_df,
                                                          class_)
        metrics[k_][class_] = protected_class_metrics
    return handle_metrics(
        metrics=metrics,
        debug=debug,
        extra_display_cols=extra_display_cols,
        suppress_output=suppress_output
    )


def calculate_class_metrics(y_true, y_pred, df, metrics_class):
    protected_class_metrics = dict(
        num_samples=sum(df[metrics_class] == 1),
        equal_odds_difference=equalized_odds_difference(
            y_true,
            y_pred,
            sensitive_features=df[metrics_class]
        ),
        equal_odds_ratio=equalized_odds_ratio(
            y_true,
            y_pred,
            sensitive_features=df[metrics_class]
        ),
        demographic_parity_difference=demographic_parity_difference(
            y_true,
            y_pred,
            sensitive_features=df[metrics_class]),
        demographic_parity_ratio=demographic_parity_ratio(
            y_true,
            y_pred,
            sensitive_features=df[metrics_class]
        ),
    )
    return protected_class_metrics


def format_metrics(metrics: dict) -> dict:
    """
    Format metrics for display by limiting precision to 3 digits.

    :param metrics:
    :return:
    """
    formatted = dict()
    for k, v in metrics.items():
        if isinstance(v, dict):
            formatted[k] = format_metrics(v)
        elif isinstance(v, float):
            formatted[k] = f'{v:.3f}'
        else:
            formatted[k] = v
    return formatted


def handle_metrics(
        metrics: dict,
        debug=False,
        tablefmt="github",
        extra_display_cols: list[tuple[str, str]] = None,
        suppress_output: bool = False
):
    """
    By default, print to stdout. Optionally save output to file.

    :param tablefmt:
    :param debug:
    :param metrics:
    :param output: If not provided, print to stdout. Otherwise, save to file specified.
    :return:
    """
    formatted = format_metrics(metrics)
    extra_display_cols_names = [] if not extra_display_cols else [col for col, _ in extra_display_cols]
    extra_display_cols_vals = [] if not extra_display_cols else [val for _, val in extra_display_cols]
    top = dict()
    if debug:
        print("Model-level metrics:")
    for k, v in sorted(list(chain(zip(extra_display_cols_names, extra_display_cols_vals), formatted.items()))):
        if k.endswith('_class_metrics'):
            continue
        top[k] = formatted.get(k, v)
        if debug:
            print(f'{k}: {v}')
    columns = list()
    rows = list()

    for class_type in ('protected', 'reference'):
        k_ = f'{class_type}_class_metrics'
        if k_ not in metrics:
            continue
        if not columns:
            columns = (
                    extra_display_cols_names
                    + ["protected_or_reference", "class"]
                    + list(next(iter(formatted[k_].values())).keys())
            )
        if debug:
            print(json.dumps(formatted, indent=4, sort_keys=True, cls=NpEncoder))
        for protected_class, protected_metrics in formatted[k_].items():
            rows.append((
                    extra_display_cols_vals
                    + [class_type, protected_class]
                    + [v for k, v in protected_metrics.items()])
            )
    if not suppress_output:
        print('')
        print(
            tabulate(
                rows,
                headers=columns,
                tablefmt=tablefmt
            )
        )
        print("\n\n")
    return pd.DataFrame(rows, columns=columns), top


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, type=str, help="Path to input file")
    parser.add_argument("--protected-classes", required=True, type=str,
                        help="Comma-separated list of protected classes.")
    parser.add_argument("--reference-classes", required=True, type=str,
                        help="Comma-separated list of reference classes.")
    parser.add_argument("--binary-outcome-col", required=True, type=str,
                        help="Column containing binary outcome data on patient.")
    parser.add_argument("--probability-col", required=True, type=str, help="Probability outcome column.")
    parser.add_argument("--sample-weights-col", type=str, help="Column sample weights.")
    parser.add_argument("--pos-outcome-indicator", default="1", type=str, help="Binary outcome column positive value.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logging.")
    args = parser.parse_args()

    protected_classes = args.protected_classes.strip().split(',')
    reference_classes = args.reference_classes.strip().split(',') if args.reference_classes is not None else None

    if args.debug:
        logging.basicConfig(level="DEBUG")

    measure(
        inp=args.input_file,
        binary_outcome_column=args.binary_outcome_col,
        protected_classes=protected_classes,
        reference_classes=reference_classes,
        sample_weights_column=args.sample_weights_col,
        pos_outcome_indicator=args.pos_outcome_indicator,
        probability_column=args.probability_col,
        debug=args.debug
    )
