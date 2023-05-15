# measure_disparity.py - iAdeptive technologies
# CLI interface included, use -h for help

import configparser
import argparse
import pandas as pd
import random
import utils.measure_disparity_methods as measure_disparity_methods
import os

random.seed(42)
parser = argparse.ArgumentParser(description="iAdeptive Bias Measurement Tool",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("settings", help = "settings.ini file")
args = vars(parser.parse_args())

Config = configparser.ConfigParser()
Config.read(args['settings'])

def measure(Config):
    data = pd.read_csv(Config['paths']['input_path'])
    metrics_list = Config['other']['display_metrics'].split(',')
    demo_dict = dict(zip(Config['dataset information']['demo_variables'].split(','), 
                         Config['dataset information']['advantaged_groups'].split(',')))
    demos = measure_disparity_methods.measured(data, 
                                      Config['dataset information']['demo_variables'].split(','), 
                                      intercols = None, 
                                      actualcol=Config['dataset information']['target_variable'], 
                                      probabilitycol=Config['dataset information']['probability_column'],
                                      weightscol = Config['dataset information']['weight_column'])
    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    for key, value in demo_dict.items():
        print("-------")
        print(key)
        print("-------")
        ratio_df = demos.PrintRatios(key, value, metrics= metrics_list, printout=True)
        plot1 = demos.MetricPlots(key, value, metrics = metrics_list, draw=True)
        plot1.save('graphs/' + str(key) + '_disparity.png')

measure(Config)