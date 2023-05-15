# mitigate_disparity.py - iAdeptive technologies
# CLI interface included, use -h for help

import configparser
import argparse
import pandas as pd
import random
from utils import mitigate_disparity_methods
import utils.measure_disparity_methods as measure_disparity_methods
import os

random.seed(42)
parser = argparse.ArgumentParser(description="iAdeptive Bias Mitigation Tool",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("settings", help = "settings.ini file")
parser.add_argument("-a", "-adjust", action="store_true", help = "Model Agnostic Adjustment")
parser.add_argument("-f", "-fit", action="store_true", help = "Fit Debiased Model Thresholds")
parser.add_argument("-p", "-predict", action="store_true", help = "Fit Debiased Model and Predict")
args = vars(parser.parse_args())

Config = configparser.ConfigParser()
Config.read(args['settings'])

def adjust_option(Config):
    adjustment_data = pd.read_csv(Config['paths']['input_path'])
    y = adjustment_data[Config['dataset information']['target_variable']]
    D = adjustment_data[Config['dataset information']['demo_variables'].split(',')]
    proba = adjustment_data[Config['dataset information']['probability_column']]
    demo_dict = dict(zip(Config['dataset information']['demo_variables'].split(','),
                         Config['dataset information']['advantaged_groups'].split(',')))
    granularity = float(Config['other']['debias_granularity'])

    reps = int(Config['other']['optimization_repetitions'])
    adjusted = mitigate_disparity_methods.model_agnostic_adjustment(adjustment_data, y, proba, D, demo_dict, reps, granularity)
    adjusted.to_csv(Config['paths']['output_path'], index = False)

def fit_option(Config):
    modeling_data = pd.read_csv(Config['paths']['input_path'])
    X = modeling_data[Config['dataset information']['input_variables'].split(',')]
    y = modeling_data[Config['dataset information']['target_variable']]
    D = modeling_data[Config['dataset information']['demo_variables'].split(',')]
    demo_dict = dict(zip(Config['dataset information']['demo_variables'].split(','),
                         Config['dataset information']['advantaged_groups'].split(',')))

    db_model = mitigate_disparity_methods.debiased_model(Config)

    db_model.fit(X, y, D, demo_dict)
    print(db_model.debiased_thresholds)

def predict_option(Config):
    modeling_data = pd.read_csv(Config['paths']['input_path'])
    X = modeling_data[Config['dataset information']['input_variables'].split(',')]
    y = modeling_data[Config['dataset information']['target_variable']]
    D = modeling_data[Config['dataset information']['demo_variables'].split(',')]
    demo_dict = dict(zip(Config['dataset information']['demo_variables'].split(','),
                         Config['dataset information']['advantaged_groups'].split(',')))

    predict_data = pd.read_csv(Config['paths']['predict_input_path'])
    P = predict_data[Config['dataset information']['input_variables'].split(',')]

    db_model = mitigate_disparity_methods.debiased_model(Config)

    db_model.fit(X, y, D, demo_dict)

    results = db_model.predict(P, D)
    predict_data['predicted'] = results

    predict_data.to_csv(Config['paths']['output_path'], index = False)

    demos = measure_disparity_methods.measured(predict_data, 
                                      Config['dataset information']['demo_variables'].split(','), 
                                      intercols = None, 
                                      actualcol=Config['dataset information']['target_variable'], 
                                      probabilitycol=Config['dataset information']['probability_column'],
                                      weightscol = Config['dataset information']['weight_column'])
    
    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    for key, value in demo_dict.items():
        ratio_df = demos.PrintRatios(key, value, printout=True)
        plot1 = demos.MetricPlots(key, value, draw=True)
        plot1.save('graphs/' + str(key) + '_disparity.png')

if ('a' in args) and (args['a'] == True):
    adjust_option(Config)
if ('f' in args) and args['f'] == True:
    fit_option(Config)
if ('p' in args) and args['p'] == True:
    predict_option(Config)
