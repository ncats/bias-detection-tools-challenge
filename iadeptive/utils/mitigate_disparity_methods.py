# mitigate_disparity_methods.py - iAdeptive technologies

import pandas as pd
import numpy as np
import random
import importlib

np.seterr(divide='ignore')

def _cutoff_truths(actual, proba, cutoff):
    """
    _cutoff_truths
    Calculates truth values for a specified cutoff

    actual [pd.Dataframe]: actual values
    proba [pd.Dataframe]: model probability values
    cutoff [float]: cutoff for positive label
    """
    predicted = (proba > cutoff)
    truths = actual + predicted * 2

    truths = (truths
              .replace(3, 'TP')
              .replace(2, 'FP')
              .replace(1, 'FN')
              .replace(0, 'TN'))
    return truths

def _map_cutoff_ranges(actual, proba, thresholds, granularity, baseline = False):
    """
    _map_cutoff_ranges
    calculates truth values for various cutoff ranges 

    actual [pd.Dataframe]: actual values
    proba [pd.Dataframe]: model probability values
    thresholds [pd.Dataframe] thresholds tracker passthrough
    granularity [float] debiasing granularity
    baseline [bool, default = False]: Bool for whether to keep thresholds at 0.5 for advantaged group
    """
    range = pd.Series(np.arange(0.1, 0.9, granularity))
    if baseline == False:
        truth_matrix = range.apply(lambda x: _cutoff_truths(actual,proba, x)).transpose()
        truth_matrix.columns = np.arange(0.1, 0.9, granularity)
    else:
        truth_matrix = range.apply(lambda x: _cutoff_truths(actual, proba, thresholds)).transpose()
        truth_matrix.columns = np.arange(0.1, 0.9, granularity)

    truth_summary = pd.DataFrame()
    TP = (truth_matrix == 'TP').sum( axis = 0)
    TN = (truth_matrix == 'TN').sum( axis = 0)
    FP = (truth_matrix == 'FP').sum( axis = 0)
    FN = (truth_matrix == 'FN').sum( axis = 0)

    truth_summary = pd.DataFrame()
    truth_summary['STP'] = (TP + FP) / (TP + FP + TN + FN)
    truth_summary['ACC'] = (TP + TN) / (TP + TN + FP + FN)
    truth_summary['TPR'] = (TP) / (TP + FN)
    truth_summary['PPV'] = (TP) / (TP + FP)
    truth_summary['FPR'] = (FP) / (FP + TN)
    
    return truth_summary

def _map_ratios(actual, proba, threshold, demo, advantaged_label, granularity):
    """
    _map_ratios
    maps bias ratios across different thresholds for a particular demographic

    actual [pd.Dataframe]: actual values
    proba [pd.Dataframe]: model probability values
    threshold [pd.Dataframe] thresholds tracker passthrough
    demo [str]: demographic label,
    advantaged_label [str:] advantaged group
    granularity [float] debiasing granularity
    """
    adv_actual = actual[demo == advantaged_label]
    adv_proba = proba[demo == advantaged_label]
    adv_threshold = threshold[demo == advantaged_label]
    
    adv_tm = _map_cutoff_ranges(adv_actual, 
                               adv_proba, 
                               adv_threshold, 
                               granularity, 
                               baseline = True)

    tm_dict = {}
    for d in demo.unique():
        if d == advantaged_label:
            continue
        else:
            metrics = _map_cutoff_ranges(actual[demo == d],
                                        proba[demo == d],
                                        threshold[demo == d],
                                        granularity)
        
        ratios = (metrics / adv_tm)
        ratios[ratios > 1] = 1
        ratios = abs(np.log(ratios))

        ratios.columns = metrics.columns + '_loss'
        combined = pd.concat([metrics, ratios], axis = 1)
        combined = combined.reset_index()
        combined['demo_label'] = d
        combined = combined.rename(columns = {'index':'threshold'})
        tm_dict[d] = combined

    output = pd.concat(tm_dict.values()).replace([np.inf, -np.inf], np.nan).dropna()
    return output

def _optimize_thresholds(actual,
                         proba,
                         threshold,
                         demo,
                         advantaged_label,
                         granularity):
    """
    _optimize_thresholds
    optimizes thresholds across a particular demographic group

    actual [pd.Dataframe]: actual values
    proba [pd.Dataframe]: model probability values
    threshold [pd.Dataframe] thresholds tracker passthrough
    demo [str]: demographic label,
    advantaged_label [str:] advantaged group
    granularity [float] debiasing granularity
    """
    ratios = _map_ratios(actual, proba, threshold,  demo, advantaged_label, granularity)
    ratios['normalized_loss_score'] = ratios[[col for col in ratios.columns if col.endswith('_loss')]].apply(lambda x: (x - x.mean())/x.std(), axis = 0).sum(axis = 1)
    top = ratios.sort_values(by= ['demo_label', 'normalized_loss_score'], ascending = True).groupby('demo_label').head(1)[['demo_label','threshold']]

    output = pd.DataFrame()
    output['demo_label'] = demo

    top = pd.concat([top, pd.DataFrame({'demo_label':[advantaged_label], 'threshold': [0.5]})])
    output = output.merge(top, on = 'demo_label', how = 'left').threshold.values

    return output

def _return_optimized_thresholds(actual,
                                proba,
                                demo_data,
                                demo_dict,
                                reps,
                                granularity,
                                thresholds_only = False):
    """
    _return_optimized_thresholds
    returns optimized thresholds across all demographic groups designated

    actual [pd.Dataframe]: actual values
    proba [pd.Dataframe]: model probability values
    threshold [pd.Dataframe] thresholds tracker passthrough
    demo_data [pd.Dataframe]: demographic data columns,
    demo_dict [dictionary {str:str}]: a dict of demographic column names paired with their respective advanteged group
    reps [int]: number of debiasing repetitions
    granularity [float] debiasing granularity
    thresholds_only [bool, default = False]: Bool for whether to return a full updated dataset or new demo cutoffs only
    """
        
    i = 0
    thresholds = np.repeat(0.5, len(proba))
    tracking_df = pd.DataFrame()
    tracking_df[i] = thresholds
    choice = random.choice(list(demo_dict.keys()))
    
    for n in range(1,reps+1):
        print("Debiasing iteration " + str(n)+"/"+str(reps))
        demo = choice 
        results = _optimize_thresholds(actual,
                                       proba,
                                       thresholds,
                                       demo_data[demo],
                                       demo_dict[demo],
                                       granularity)
        thresholds = (thresholds + results) / 2
        i += 1
        tracking_df[i] = thresholds
        new_demo = demo_dict.copy()
        new_demo.pop(choice)

        choice = random.choice(list(new_demo.keys()))
    
    output = demo_data.copy()
    output['threshold'] = tracking_df.iloc[:,len(tracking_df.columns)-5:len(tracking_df.columns)].mean(axis = 1)

    if thresholds_only == False:
        output = output.drop_duplicates()
    else:
        output = output['threshold']
        
    return output

def model_agnostic_adjustment(modeling_dataset,
                              y,
                              proba,
                              D,
                              demo_dict,
                              reps = 30,
                              granularity = 0.01):
    """
    model_agnostic_adjustment
    Conducts a data and model agnostic debiasing adjustment
    
    modeling_dataset [pd.Dataframe]: modeling dataset
    y [pd.Dataframe]: target column
    proba [pd.Dataframe]: model probability values
    D [pd.Dataframe]: demographic columns
    demo_dict [dictionary {str:str}]: a dict of demographic column names paired with their respective advanteged group
    reps [int]: number of debiasing repetitions
    granularity [float] debiasing granularity
    """
    thresholds = _return_optimized_thresholds(y, 
                                              proba,
                                              D,
                                              demo_dict,
                                              reps,
                                              granularity,
                                              thresholds_only = True)
    output = modeling_dataset.copy()
    output['debiased_prediction'] = 0 
    output.loc[proba > thresholds, 'debiased_prediction'] = 1
    
    return output

class debiased_model:
    """
    debiased_model
    debiased model object

    Usage

    [new object] = debiased_model(config)
    Config: settings.ini file
    
    debiased_model.fit(X, y, D, *args, **kwargs)
    X [pd.Dataframe]: input columns
    y [pd.Dataframe]: target column
    D [pd.Dataframe]: demographic columns
    *args, **kwargs: optional argument passthroughs to connected module

    debiased_model.predict(X, D, *args, **kwargs)
    X [pd.Dataframe]: input columns
    D [pd.Dataframe]: demographic columns
    *args, **kwargs: optional argument passthroughs to connected module
    """
    def __init__(self, config):
        self.model_settings = dict(config['model settings'])
        self.model_arguments = dict(config['model arguments'])
        for key, value in self.model_arguments.items():
            try:
                self.model_arguments[key] = int(value)
            except Exception:
                try:
                    self.model_arguments[key] = float(value)
                except Exception:
                    pass
        
        self.debiased_thresholds = pd.DataFrame()

        self.reps = int(config['other']['optimization_repetitions'])
        self.granularity = float(config['other']['debias_granularity'])
        self.dataset_info = dict(config['dataset information'])

        self.model_module = importlib.import_module(self.model_settings['model_module'])

        model_class = getattr(self.model_module, self.model_settings['model_class'])

        self.model = model_class(**self.model_arguments)

        self.predict_proba = getattr(self.model, self.model_settings['predict_proba_method'])

    def fit(self, X, y, D, demo_dict, *args, **kwargs):
        fit_method = getattr(self.model, self.model_settings['fit_method'])
        fit_method(X, y, *args, **kwargs)

        proba_results = self.model.predict_proba(X)[:,1]

        self.debiased_thresholds = _return_optimized_thresholds(y, proba_results, D, demo_dict, self.reps, self.granularity)

    def predict(self, X, D, *args, **kwargs):
        proba_values = self.predict_proba(X)[:,1]
        probabilities = pd.DataFrame({"proba": proba_values})

        prediction_df = pd.concat([D, probabilities], axis = 1)
        prediction_df = prediction_df.merge(self.debiased_thresholds, on = list(D.columns), how = 'left')
        prediction_df['predicted'] = 0
        prediction_df.loc[prediction_df['proba'] > prediction_df['threshold'], 'predicted'] = 1

        return prediction_df[['predicted']]
    
    def get_debiased_thresholds(self):
        return self.debiased_thresholds