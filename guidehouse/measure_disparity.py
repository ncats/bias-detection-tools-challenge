import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score ,balanced_accuracy_score,roc_auc_score ,f1_score
from fairlearn.metrics import MetricFrame,selection_rate, true_positive_rate,false_positive_rate
from fairlearn.experimental.enable_metric_frame_plotting import plot_metric_frame

# custom function to measure fairness :

def display_fairness_metrics(path_to_df,filename,ground_truth_column_name,predictions_column_name,sensitive_feature):

    """"
    A function to display social bias metrics along with the model performance metrics:
    path_to_df: the path to where the dataframe that has the model's predictions , ground truth labels, and sensitive feature values is stored
    filename: the name of the dataframe above without .csv 
    ground_truth_column_name: the ground truth column in the test data
    predictions_column_name: the model's binary predictions on the test data
    sensitive_feature: the variable(s) we identify as sensitive/protected
    """
   
    df=pd.read_csv(path_to_df)

    y_test=df[ground_truth_column_name]
    preds=df[predictions_column_name]
    sensitive_feature=df[sensitive_feature]

    metrics={'demographic parity':selection_rate,
    'true positive rate': true_positive_rate,
    'false positive rate': false_positive_rate,
    'accuracy': accuracy_score,
    'balanced accuracy':balanced_accuracy_score,
    'f1 score': f1_score,
    'AUC': roc_auc_score}

    metric_frame=MetricFrame(metrics=metrics,y_true=y_test,y_pred=preds,sensitive_features=sensitive_feature)

    fairness_metrics=metric_frame.by_group.to_csv(f'outputs/{filename}_Fairness_Scores.csv')

    fig=plot_metric_frame(metric_frame,kind='bar', subplots=True,legend=False ,figsize=[6,18] , title=f'{filename}/fairness and performance metrics')
    plt.savefig(f'outputs/{filename}_Fairness_Performance_Metrics.png')




# computing the performance and fairness metrics by subgroup of sensitive/protected feature:

display_fairness_metrics('inputs/pre_debiasing_predictions.csv','pre_debiasing_predictions','true_label','predictions','race_ethnicity')
