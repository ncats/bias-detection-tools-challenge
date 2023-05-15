# -*- coding: utf-8 -*-
import sys
import os
from os import path
from dataclasses import dataclass

import toml
import pandas as pd
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split


@dataclass(eq=True, frozen=True)
class FairnessColumn:
    prot_col: str
    ref_col: str
    group: str


@dataclass
class Dataset:
    df: pd.DataFrame
    fairness_cols: list
    outcome_col: str
    label_col: str
    dataset_name: str

    def __post_init__(self):
        for col in self.fairness_cols:
            for col_name in [col.prot_col, col.ref_col]:
                if col_name not in self.df.columns:
                    raise ValueError('column %s not found in the input data'
                                     % col_name)

    def from_toml(toml_file: str, csv_type=None) -> 'Dataset':
        with open(toml_file, 'r') as f:
            d = toml.load(f)
        csv_file = d['dataset_csv']
        if csv_type is not None:
            if not isinstance(csv_file, dict):
                raise ValueError('if csv_type is not None, dataset_csv in '
                                 'toml configuration file should be a dict')
            csv_file = csv_file[csv_type]
        df = pd.read_csv(csv_file)
        fairness_cols = [FairnessColumn(*x) for x in d['fairness_cols']]
        return Dataset(df, fairness_cols=fairness_cols,
                       outcome_col=d.get('outcome_col', 'y_hat'),
                       label_col=d['label_col'],
                       dataset_name=d['dataset_name'])


def load_hmda(num_sample=10000, random_state=None) -> Dataset:
    dir_name = os.path.dirname(sys.modules[__name__].__file__)
    df = pd.read_csv(path.join(dir_name, 'data/hmda.csv.gz'), index_col='id')
    df['Deserved'] = df['Low-Priced']
    df.sample(n=num_sample, random_state=random_state)
    fairness_cols = [FairnessColumn('Black', 'White', 'Race')]
    return Dataset(df, fairness_cols, outcome_col='Low-Priced',
                   label_col='Deserved', dataset_name='HMDA')


def get_random_dataset(n_sample=1000, random_state=None) -> Dataset:
    df = pd.DataFrame(np.random.randint(0, 2, size=(100, 3)),
                      columns=['y', 'y_hat', 'Black'])
    # df['y_hat'] = 0
    # df.index.name = 'row_id'
    df['White'] = df['Black'].apply(lambda x: int(not x))
    fairness_cols = [FairnessColumn('Black', 'White', 'Race')]
    return Dataset(df, fairness_cols, outcome_col='y_hat',
                   label_col='y', dataset_name='random_dataset')


def get_random_train_test(n_samples=1000, n_features=10,
                          random_state=None, test_size=.33) -> tuple:
    X, y = make_gaussian_quantiles(
        cov=2.0, n_samples=n_samples, n_features=n_features, n_classes=2,
        random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    cols = ['f%d' % i for i in range(n_features - 2)] + ['Black', 'White']
    X_train = pd.DataFrame(X_train, cols=cols)
    X_train['White'] = X_train['Black'].apply(lambda x: int(not x))
    X_test = pd.DataFrame(X_test, cols=cols)
    X_test['White'] = X_test['Black'].apply(lambda x: int(not x))

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    ds = get_random_dataset()
    ds.df.to_csv('example_detection.csv')

    pass
