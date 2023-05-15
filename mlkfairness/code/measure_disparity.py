# -*- coding: utf-8 -*-
import argparse
from argparse import RawTextHelpFormatter

from mlkfairness.detector import BiasDetector
from mlkfairness.data import Dataset

__mydebug__ = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect bias in ML models.',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('dataset_config', type=str,
                        help='''Path to a TOML description of the dataset.
Example configuration file:
dataset_name = 'Example dataset'

# Path to the CSV file of the dataset.
dataset_csv = 'example_detection.csv'

# Name of the column which stores actual labels. Actual labels
#   are 0 or 1 Boolean values.
label_col = 'y'

# Name of the column which stores predicted labels. Predicated labels
#   are 0 or 1 Boolean values.
outcome_col = 'y_hat'

# Names of the columns which store protected and reference
#   characteristcs (e.g., Black and White) and the category of
#   the characteristic (Race). Multiple such triples can be provided.
#   The columns are 0 or 1 Boolean values.
# fairness_cols = [['Black', 'White', 'Race'], ['Asian', 'White', 'Race']]
fairness_cols = [['Black', 'White', 'Race']]

# [Optional] Name of the column that stores record timestamps. Column
#   values should be strings in 'dd/mm/YYYY HH:MM:SS' format.
time_col = 'Time' ''')

    if __mydebug__:
        args = parser.parse_args(
            'example_detection.toml'.split())
        # args = parser.parse_args('--run-inside'.split())
    else:
        args = parser.parse_args()

    ds = Dataset.from_toml(args.dataset_config)

    print(ds.df.info())
    adetector = BiasDetector(ds)
    ret = adetector.generate_report()
