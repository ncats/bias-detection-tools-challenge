# -*- coding: utf-8 -*-
from mlkfairness.data import Dataset
from mlkfairness.detector import BiasDetector
from mlkfairness.mitigator import FABulous
import argparse
from argparse import RawTextHelpFormatter


__mydebug__ = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mitigate bias in ML models.',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('dataset_config', type=str,
                        help='''Path to a TOML description of the training'''
                        '''and test datasets.
Example configuration file:
dataset_name = 'Example dataset'

# Path to the CSV file of the training and test dataset.
dataset_csv = {'train' = 'example_train.csv', 'test' = 'example_test.csv'}

# Name of the column which stores actual labels. Actual labels
#   are 0 or 1 Boolean values.
label_col = 'y'

# Names of the columns which store protected and reference
#   characteristcs (Black and White) and the category of
#   the characteristic (Race). Multiple such triples can be provided.
#   The columns are 0 or 1 Boolean values.
# fairness_cols = [['Black', 'White', 'Race'], ['Asian', 'White', 'Race']]
fairness_cols = [['Black', 'White', 'Race']] ''')

    if __mydebug__:
        args = parser.parse_args('example_mitigation.toml'.split())
    else:
        args = parser.parse_args()

    ds_train = Dataset.from_toml(args.dataset_config, 'train')
    X_train = ds_train.df.copy(deep=True).drop(ds_train.label_col, axis=1)
    y_train = ds_train.df[ds_train.label_col].copy(deep=True)

    ds_test = Dataset.from_toml(args.dataset_config, 'test')
    X_test = ds_test.df.copy(deep=True).drop(ds_test.label_col, axis=1)
    y_test = ds_test.df[ds_test.label_col].copy(deep=True)

    amitigator = FABulous(prot_col='Black', ref_col='White',
                          fairness_criteria='get_equal_opportunity')
    amitigator.fit(X_train, y_train)
    y_pred = amitigator.predict(X_test)
    ds_test.df['y_hat'] = y_pred
    adetector = BiasDetector(ds_test)
    ret = adetector.generate_report()
