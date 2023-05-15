#!/usr/bin/env python
# coding: utf-8

# In[1]:


################
## USER INPUT ##
################

# CRITICAL: Fill csv file name for the model output to be measured
model_csv = 'input_model.csv'

# CRITICAL: Enter column names as a list of each demographic feature (as strings) you want to measure social fairness for
protected_features = ['age','race','gender']

# CRITICAL: Enter column name of model prediction probability
pred_prob = 'prediction_probability'

# CRITICAL: Enter column name of true label value
y_bar = 'readmitted'

# CRITICAL: Enter column name containing sample weights
samp_weight = 'sw'

###########
## SETUP ##
###########

import platform
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import statsmodels.stats.proportion as ssp
from itertools import combinations
from itertools import permutations
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import chi2, chisquare
import statistics
from plotly import express as px
from math import asin, sqrt

date=datetime.today().strftime('%Y-%m-%d')

# Define OS
try: 
    op_sys = platform.system()
except Exception as e:
    print('Error:', e)

# Define Model Paths for OS
in_directory_linux_unix = r'../input_model'
in_directory_windows = r'..\input_model'

# Definie Report Paths for OS
out_directory_linux_unix = r'../reports'
out_directory_windows = r'..\reports'

###########################
## LOAD AND PREPARE DATA ##
###########################

# Load data into a dataframe
if op_sys == 'Windows':
    input_df = pd.read_csv(os.path.join(in_directory_windows, model_csv), header=0, 
                           encoding='utf-8')
elif op_sys == 'Linux' or op_sys == 'Darwin':
    input_df = pd.read_csv(os.path.join(in_directory_linux_unix, model_csv), header=0, 
                           encoding='utf-8')   
else: print('Error: Unknown OS!')
            
# Use ROC curve to set optimal model prediction threshold and create y_hat for binary prediction value
fpr_roc, tpr_roc, threshold_positive = metrics.roc_curve(input_df[y_bar], input_df[pred_prob])
full_roc = pd.DataFrame(zip(fpr_roc, tpr_roc, threshold_positive), columns = ['fpr','tpr','thrsh'])
full_roc['rate_diff'] = full_roc['tpr'] - full_roc['fpr']
threshold_positive = full_roc.loc[full_roc.rate_diff.idxmax(), 'thrsh']

input_df.loc[input_df[pred_prob] > threshold_positive, 'y_hat']=int(1)
input_df[['y_hat']] = input_df[['y_hat']].fillna(value=0)
            

###############
## FUNCTIONS ##
###############

def MetricsReport(y_bar, y_hat, group_type, group_name, samp_weight):
    dict_metrics = {}
    dict_metrics['Protected Group Type'] = group_type        
    dict_metrics['Protected Group'] = group_name

            
    # Confusion Matrix
    tn, fp, fn, tp = metrics.confusion_matrix(y_bar,y_hat, labels=[0,1], sample_weight = samp_weight).ravel()
    dict_metrics['TN'] = '{:.0f}'.format(tn)
    dict_metrics['FP'] = '{:.0f}'.format(fp)
    dict_metrics['FN'] = '{:.0f}'.format(fn)
    dict_metrics['TP'] = '{:.0f}'.format(tp)
    
    # Number of Observations
    obs_count = tp+fp+tn+fn
    dict_metrics['Observation Count'] = '{:.0f}'.format(obs_count)
            
    # Demographic Parity
    dem_par = (tp+fp)/(tp+fp+tn+fn)
    dict_metrics['Demographic Parity'] = '{:.5f}'.format(dem_par)   
           
    # Precision
    prec = tp/(tp+fp)
    dict_metrics['Precision'] = '{:.5f}'.format(prec)
            
    # Equalized Opportunity
    eq_op = (tp)/(tp+fn)
    dict_metrics['Equalized Opportunity'] = '{:.5f}'.format(eq_op)
            
    # Social Fairness
    social = statistics.harmonic_mean([dem_par,eq_op])
    dict_metrics['Social Parity Score'] = '{:.5f}'.format(social)
    
    # KS Statistic
    fpr = fp/(fp+tn)
    ks = eq_op-fpr
    dict_metrics['KS Statistic'] = '{:.5f}'.format(ks)
    
    return dict_metrics
                  
            
#######################
## CALCULATE METRICS ##
#######################            
            
# Calculate metrics for dataset-wide observations             
total_model_results=MetricsReport(list(zip(input_df[y_bar].tolist())),
                                  list(zip(input_df.y_hat.tolist())),
                                  'All Observations',
                                  'All Observations',
                                  input_df[samp_weight])

# Create dataframe to store metrics
metrics_df = pd.DataFrame([total_model_results])
            
# Add metrics for each protected group within the protected_features    

for feature in protected_features:   
    feature_values = input_df[feature].unique().tolist()
    for val in feature_values:
        results = MetricsReport(list(zip(input_df[y_bar].loc[input_df[feature]==val].tolist())),
                                list(zip(input_df.loc[input_df[feature]==val].y_hat.tolist())),
                                feature,
                                val,
                                input_df[samp_weight].loc[input_df[feature]==val])
        metrics_df = metrics_df.append(results, ignore_index=True)

# Format metrics
metrics_df['TP'] = metrics_df['TP'].astype('int64')
metrics_df['FP'] = metrics_df['FP'].astype('int64')
metrics_df['TN'] = metrics_df['TN'].astype('int64')
metrics_df['FN'] = metrics_df['FN'].astype('int64')
metrics_df['Observation Count'] = metrics_df['Observation Count'].astype('int64')
metrics_df['Demographic Parity'] = metrics_df['Demographic Parity'].astype('float64')
metrics_df['Precision'] = metrics_df['Precision'].astype('float64')
metrics_df['Equalized Opportunity'] = metrics_df['Equalized Opportunity'].astype('float64')
metrics_df['Social Parity Score'] = metrics_df['Social Parity Score'].astype('float64')
metrics_df['KS Statistic'] = metrics_df['KS Statistic'].astype('float64') 
            
# Score Differences
group_metrics = metrics_df[metrics_df['Protected Group Type'] != 'All Observations']
disparity = pd.DataFrame(columns=['Protected Group','Social Disparity Score'])
types_lst = group_metrics['Protected Group Type'].unique().tolist()                       
for cat in types_lst:
    max_score = group_metrics['Social Parity Score'].loc[group_metrics['Protected Group Type']==cat].max()
    scoring_df = group_metrics[['Protected Group','Social Parity Score']].loc[group_metrics['Protected Group Type']==cat]
    scoring_df = scoring_df.reset_index()
    for index,row in scoring_df.iterrows():
        group = row[1]
        score = row[2]
        disparity.loc[len(disparity)] = [group, max_score-score]
            
disparity_df = pd.merge(group_metrics, disparity, how='inner',
                        left_on=['Protected Group'],
                        right_on=['Protected Group'])

disparity_df['Protected Group'] = disparity_df['Protected Group'].astype(str)
disparity_df['Observation Count'] = disparity_df['Observation Count'].astype('float64')
            
# Write sorted table into HTML
disparity_table = disparity_df[['Protected Group Type','Protected Group','Social Disparity Score']]\
    .sort_values(by=['Protected Group Type','Social Disparity Score'], ascending = [True, True])\
    .to_html(classes = 'table', index=False)

            
######################            
## Chi-Squared Tests##
######################  

# Group Demographic Parity 
group_dp_chi = pd.DataFrame(columns=['Protected Group Type', 'Chi-Squared', 'P-Value'])       
for cat in protected_features:
    cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
    (chi2, p, arr) = ssp.proportions_chisquare(count=(cat_df.TP + cat_df.FP), nobs=(cat_df['Observation Count'])) 
    result = [cat, chi2, p]
    group_dp_chi.loc[len(group_dp_chi)] = result
group_dp_chi['Fail'] = group_dp_chi['P-Value']<0.01
group_dp_chi = group_dp_chi.to_html(classes='table', index=False)


# Group Equalized Opportunity           
group_eo_chi = pd.DataFrame(columns=['Protected Group Type', 'Chi-Squared', 'P-Value'])       
for cat in protected_features:
    cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
    (chi2, p, arr) = ssp.proportions_chisquare(count=cat_df.TP, nobs=(cat_df.TP + cat_df.FN)) 
    result = [cat, chi2, p]
    group_eo_chi.loc[len(group_eo_chi)] = result
group_eo_chi['Fail'] = group_eo_chi['P-Value']<0.01
group_eo_chi = group_eo_chi.to_html(classes='table', index=False)     

# Group Precision Parity 
group_prec_chi = pd.DataFrame(columns=['Protected Group Type', 'Chi-Squared', 'P-Value'])       
for cat in protected_features:
    cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
    (chi2, p, arr) = ssp.proportions_chisquare(count=(cat_df.TP), nobs=(cat_df.TP + cat_df.FP)) 
    result = [cat, chi2, p]
    group_prec_chi.loc[len(group_prec_chi)] = result
group_prec_chi['Fail'] = group_prec_chi['P-Value']<0.01
group_prec_chi = group_prec_chi.to_html(classes='table', index=False)

# Group KS Statistic Proportions 80% Rule of Thumb (Not Chi-Square)
group_ks_80 = pd.DataFrame(columns=['Protected Group Type','KS Proportion'])
for cat in protected_features:
    cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
    prop = cat_df['KS Statistic'].loc[cat_df['KS Statistic'].idxmin()]/cat_df['KS Statistic'].loc[cat_df['KS Statistic'].idxmax()]
    result = [cat, prop]
    group_ks_80.loc[len(group_ks_80)] = result
group_ks_80['Meaningful Disparity'] = group_ks_80['KS Proportion']<0.8
group_ks_80 = group_ks_80.to_html(classes='table', index=False)

# Paired Group Chi-Squared Test for Demographic Parity
dp_hm_dict = {}
for cat in protected_features:
    cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
    all_permutations = list(permutations(cat_df['Protected Group'], 2))
    cols = ['Group 1','Group 2', 'Corrected P-Value', 'Reject']
    entries = []
    p_vals = []
    for perm in all_permutations:
        perm_df = cat_df[(cat_df['Protected Group'] == perm[0]) | (cat_df['Protected Group'] == perm[1])]
        chi2, p, arr = ssp.proportions_chisquare(count = (perm_df['TP']+perm_df['FP']), 
                                                 nobs = perm_df['Observation Count'])
        p_vals.append(p)
    if cat_df.shape[0] > 2:
        reject_list, corrected_p_vals = multipletests(p_vals, alpha=0.01, method='fdr_bh')[:2]
        for perm, corr_p_val, reject in zip(all_permutations, corrected_p_vals, reject_list):
            entries.append([perm[0], perm[1], corr_p_val, reject])
    else:
        reject_list = [x < 0.01 for x in p_vals]
        for perm, corr_p_val, reject in zip(all_permutations, p_vals, reject_list):
            entries.append([perm[0], perm[1], corr_p_val, reject]) 
    df_results = pd.DataFrame(entries, columns=cols)
    df_results['Group 1'] = df_results['Group 1'].astype('str')
    df_results['Group 2'] = df_results['Group 2'].astype('str')
    df_results['Corrected P-Value'] = df_results['Corrected P-Value'].astype('float64')
    df_results['Reject'] = df_results['Reject'].astype('bool')
    results_pivot = df_results.pivot_table(index='Group 1', columns='Group 2', 
                                           values='Corrected P-Value', aggfunc='mean', fill_value=False)
    results_pivot.astype(float)
    heat_map = px.imshow(results_pivot,
                         labels=dict(x='Group 1', y='Group 2', color='P-Value'),
                         x=list(df_results['Group 1'].unique()),
                         y=list(df_results['Group 1'].unique()),
                         zmin=0, zmax=1,
                         title = 'Demographic Parity Paired Chi Square Test for ' + str(cat))
    hm_html = heat_map.to_html(full_html=False, include_plotlyjs='cdn')
    dp_hm_dict[str(cat)] = hm_html            
            
            
            
# Paired Group Chi-Squared Test for Equalized Opportunity
eo_hm_dict = {}
for cat in protected_features:
    cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
    all_permutations = list(permutations(cat_df['Protected Group'], 2))
    cols = ['Group 1','Group 2', 'Corrected P-Value', 'Reject']
    entries = []
    p_vals = []
    for perm in all_permutations:
        perm_df = cat_df[(cat_df['Protected Group'] == perm[0]) | (cat_df['Protected Group'] == perm[1])]
        chi2, p, arr = ssp.proportions_chisquare(count = perm_df['TP'], 
                                                 nobs = (perm_df['TP']+perm_df['FN']))
        p_vals.append(p)
    if cat_df.shape[0] > 2:
        reject_list, corrected_p_vals = multipletests(p_vals, alpha=0.01, method='fdr_bh')[:2]
        for perm, corr_p_val, reject in zip(all_permutations, corrected_p_vals, reject_list):
            entries.append([perm[0], perm[1], corr_p_val, reject])
    else:
        reject_list = [x < 0.01 for x in p_vals]
        for perm, corr_p_val, reject in zip(all_permutations, p_vals, reject_list):
            entries.append([perm[0], perm[1], corr_p_val, reject]) 
    df_results = pd.DataFrame(entries, columns=cols)
    df_results['Group 1'] = df_results['Group 1'].astype('str')
    df_results['Group 2'] = df_results['Group 2'].astype('str')
    df_results['Corrected P-Value'] = df_results['Corrected P-Value'].astype('float64')
    df_results['Reject'] = df_results['Reject'].astype('bool')
    results_pivot = df_results.pivot_table(index='Group 1', columns='Group 2', 
                                               values='Corrected P-Value', aggfunc='mean', fill_value=False)
    results_pivot.astype(float)
    heat_map = px.imshow(results_pivot,
                         labels=dict(x='Group 1', y='Group 2', color='P-Value'),
                         x=list(df_results['Group 1'].unique()),
                         y=list(df_results['Group 1'].unique()),
                         zmin=0, zmax=1,
                         title = 'Equalized Opporunity Paired Chi Square Test for ' + str(cat))
    hm_html = heat_map.to_html(full_html=False, include_plotlyjs='cdn')
    eo_hm_dict[str(cat)] = hm_html

# Paired Group Chi-Squared Test for Precision
prec_hm_dict = {}
for cat in protected_features:
    cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
    all_permutations = list(permutations(cat_df['Protected Group'], 2))
    cols = ['Group 1','Group 2', 'Corrected P-Value', 'Reject']
    entries = []
    p_vals = []
    for perm in all_permutations:
        perm_df = cat_df[(cat_df['Protected Group'] == perm[0]) | (cat_df['Protected Group'] == perm[1])]
        chi2, p, arr = ssp.proportions_chisquare(count = perm_df['TP'], 
                                                 nobs = (perm_df['TP']+perm_df['FP']))
        p_vals.append(p)
    if cat_df.shape[0] > 2:
        reject_list, corrected_p_vals = multipletests(p_vals, alpha=0.01, method='fdr_bh')[:2]
        for perm, corr_p_val, reject in zip(all_permutations, corrected_p_vals, reject_list):
            entries.append([perm[0], perm[1], corr_p_val, reject])
    else:
        reject_list = [x < 0.01 for x in p_vals]
        for perm, corr_p_val, reject in zip(all_permutations, p_vals, reject_list):
            entries.append([perm[0], perm[1], corr_p_val, reject]) 
    df_results = pd.DataFrame(entries, columns=cols)
    df_results['Group 1'] = df_results['Group 1'].astype('str')
    df_results['Group 2'] = df_results['Group 2'].astype('str')
    df_results['Corrected P-Value'] = df_results['Corrected P-Value'].astype('float64')
    df_results['Reject'] = df_results['Reject'].astype('bool')
    results_pivot = df_results.pivot_table(index='Group 1', columns='Group 2', 
                                               values='Corrected P-Value', aggfunc='mean', fill_value=False)
    results_pivot.astype(float)
    heat_map = px.imshow(results_pivot,
                         labels=dict(x='Group 1', y='Group 2', color='P-Value'),
                         x=list(df_results['Group 1'].unique()),
                         y=list(df_results['Group 1'].unique()),
                         zmin=0, zmax=1,
                         title = 'Precision Paired Chi Square Test for ' + str(cat))
    hm_html = heat_map.to_html(full_html=False, include_plotlyjs='cdn')
    prec_hm_dict[str(cat)] = hm_html

# Paired Group 80% Proportions Rule-of-Thumb test for KS Statistic (Differential Validity)
ks_hm_dict = {}
for cat in protected_features:
    cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
    all_permutations = list(permutations(cat_df['Protected Group'], 2))
    cols = ['Group 1','Group 2', 'KS Proportion', 'Meaningful Disparity']
    entries = []
    props = []
    for perm in all_permutations:
        perm_df = cat_df[(cat_df['Protected Group'] == perm[0]) | (cat_df['Protected Group'] == perm[1])]
        prop = perm_df['KS Statistic'].loc[perm_df['KS Statistic'].idxmin()]/perm_df['KS Statistic'].loc[perm_df['KS Statistic'].idxmax()]
        props.append(prop)
    disparity_list = [x < 0.8 for x in props]
    for perm, p, md in zip(all_permutations, props, disparity_list):
        entries.append([perm[0], perm[1], p, md])
    df_results = pd.DataFrame(entries, columns=cols)
    df_results['Group 1'] = df_results['Group 1'].astype('str')
    df_results['Group 2'] = df_results['Group 2'].astype('str')
    df_results['KS Proportion'] = df_results['KS Proportion'].astype('float64')
    df_results['Meaningful Disparity'] = df_results['Meaningful Disparity'].astype('bool')                       
    results_pivot = df_results.pivot_table(index='Group 1', columns='Group 2', 
                                               values='KS Proportion', aggfunc='mean', fill_value=False)                   
    heat_map = px.imshow(results_pivot,
                         labels=dict(x='Group 1', y='Group 2', color='KS Proportion'),
                         x=list(df_results['Group 1'].unique()),
                         y=list(df_results['Group 1'].unique()),
                         zmin=0, zmax=1,
                         title = 'KS Statistic Paired 80% Rule-of-Thumb Test for ' + str(cat))
    hm_html = heat_map.to_html(full_html=False, include_plotlyjs='cdn')
    ks_hm_dict[str(cat)] = hm_html                   
        
        
####################            
## Visualizations ##
####################

# All Metrics Bar Plot
grouped_metrics_df = pd.melt(metrics_df[['Protected Group', 'Protected Group Type', 
                                         'Demographic Parity', 'Precision',
                                         'Equalized Opportunity', 'KS Statistic'
                                        ]],
                             id_vars=['Protected Group', 'Protected Group Type'],
                             var_name='Metric')

all_metrics_bp_dict = {}            
for cat in protected_features:
    cat_df = grouped_metrics_df.loc[grouped_metrics_df['Protected Group Type']==cat]
    cat_df = cat_df.sort_values(by=['Protected Group','Metric'], ascending=True)
    multi_bar = px.bar(cat_df, x='Protected Group',
                    y='value', color='Metric',
                    barmode='group',
              labels={
                  'Protected Group':'Protected Group',
                  'Value':'Metric Score'},
              title='Metrics Comparison by ' + str(cat))
    multi_bar_html = multi_bar.to_html(full_html=False, include_plotlyjs='cdn')
    all_metrics_bp_dict[str(cat)] = multi_bar_html

            
            
# Individual Scatter Plots for Metric Vs Obs Count           
dp_sctr = px.scatter(disparity_df, x='Observation Count',
                     y='Demographic Parity', color='Protected Group Type',
                     hover_name='Protected Group',
                     labels={
                             'Observation Count':'Number of observations in data',
                             'Demographic Parity':'Demographic Parity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='Demographic Partiy Score Vs. Observation Count')
dp_sctr.update_traces(marker=dict(size=10))
dp_sctr_html = dp_sctr.to_html(full_html=False, include_plotlyjs='cdn')   
            
eo_sctr = px.scatter(disparity_df, x='Observation Count',
                     y='Equalized Opportunity', color='Protected Group Type',
                     hover_name='Protected Group',
                     labels={
                             'Observation Count':'Number of observations in data',
                             'Equalized Opportunity':'Equalized Opportunity Parity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='Equalized Opportunity Score Vs. Observation Count')
eo_sctr.update_traces(marker=dict(size=10))
eo_sctr_html = eo_sctr.to_html(full_html=False, include_plotlyjs='cdn')

prec_sctr = px.scatter(disparity_df, x='Observation Count',
                     y='Precision', color='Protected Group Type',
                     hover_name='Protected Group',
                     labels={
                             'Observation Count':'Number of observations in data',
                             'Precision':'Precision Parity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='Precision Score Vs. Observation Count')
prec_sctr.update_traces(marker=dict(size=10))
prec_sctr_html = prec_sctr.to_html(full_html=False, include_plotlyjs='cdn')

ks_sctr = px.scatter(disparity_df, x='Observation Count',
                     y='KS Statistic', color='Protected Group Type',
                     hover_name='Protected Group',
                     labels={
                             'Observation Count':'Number of observations in data',
                             'KS Statistic':'KS Statistic Differential Validity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='KS Statistic Differential Validity Score Vs. Observation Count')
ks_sctr.update_traces(marker=dict(size=10))
ks_sctr_html = ks_sctr.to_html(full_html=False, include_plotlyjs='cdn')
            
sds_sctr = px.scatter(disparity_df, x='Observation Count',
                     y='Social Disparity Score', color='Protected Group Type',
                     hover_name='Protected Group',
                     labels={
                             'Observation Count':'Number of observations in data',
                             'Social Disparity Score':'Group Social Disparity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='Social Disparity Score Vs. Observation Count')
sds_sctr.update_traces(marker=dict(size=10))
sds_sctr_html = sds_sctr.to_html(full_html=False, include_plotlyjs='cdn')   

#Scatter Plots for Metric vs Metric
dp_v_eo_sctr = px.scatter(disparity_df, x='Equalized Opportunity',
                     y='Demographic Parity', color='Protected Group Type',
                     hover_name='Protected Group', hover_data=['Observation Count'],
                     labels={
                             'Demographic Parity':'Demographic Parity Score',
                             'Equalized Opportunity':'Equalized Opportunity Parity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='Demographic Parity Score Vs. Equalized Opportunity Score')
dp_v_eo_sctr.update_traces(marker=dict(size=10))
dp_v_eo_sctr_html = dp_v_eo_sctr.to_html(full_html=False, include_plotlyjs='cdn')  

dp_v_prec_sctr = px.scatter(disparity_df, x='Precision',
                     y='Demographic Parity', color='Protected Group Type',
                     hover_name='Protected Group', hover_data=['Observation Count'],
                     labels={
                             'Demographic Parity':'Demographic Parity Score',
                             'Precision':'Precision Parity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='Demographic Parity Score Vs. Precision Score')
dp_v_prec_sctr.update_traces(marker=dict(size=10))
dp_v_prec_sctr_html = dp_v_prec_sctr.to_html(full_html=False, include_plotlyjs='cdn')
                       
dp_v_ks_sctr = px.scatter(disparity_df, x='KS Statistic',
                     y='Demographic Parity', color='Protected Group Type',
                     hover_name='Protected Group', hover_data=['Observation Count'],
                     labels={
                             'Demographic Parity':'Demographic Parity Score',
                             'KS Statistic':'KS Statistic Differential Validity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='Demographic Parity Score Vs. KS Statistic')
dp_v_ks_sctr.update_traces(marker=dict(size=10))
dp_v_ks_sctr_html = dp_v_ks_sctr.to_html(full_html=False, include_plotlyjs='cdn')   

eo_v_prec_sctr = px.scatter(disparity_df, x='Precision',
                     y='Equalized Opportunity', color='Protected Group Type',
                     hover_name='Protected Group', hover_data=['Observation Count'],
                     labels={
                             'Equalized Opportunity':'Equalized Opportunity Score',
                             'Precision':'Precision Parity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='Equalized Opportunity Score Vs. Precision Score')
eo_v_prec_sctr.update_traces(marker=dict(size=10))
eo_v_prec_sctr_html = eo_v_prec_sctr.to_html(full_html=False, include_plotlyjs='cdn')

eo_v_ks_sctr = px.scatter(disparity_df, x='KS Statistic',
                     y='Equalized Opportunity', color='Protected Group Type',
                     hover_name='Protected Group', hover_data=['Observation Count'],
                     labels={
                             'Equalized Opportunity':'Equalized Opportunity Score',
                             'KS Statistic':'KS Statistic Differential Validity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='Equalized Opportunity Score Vs. KS Statistic')
eo_v_ks_sctr.update_traces(marker=dict(size=10))
eo_v_ks_sctr_html = eo_v_ks_sctr.to_html(full_html=False, include_plotlyjs='cdn')

prec_v_ks_sctr = px.scatter(disparity_df, x='KS Statistic',
                     y='Precision', color='Protected Group Type',
                     hover_name='Protected Group', hover_data=['Observation Count'],
                     labels={
                             'Precision':'Precision Parity Score',
                             'KS Statistic':'KS Statistic Differential Validity Score',
                             'Protected Group Type':'Protected Group Type'},
                     title='Precision Parity Score Vs. KS Statistic')
prec_v_ks_sctr.update_traces(marker=dict(size=10))
prec_v_ks_sctr_html = prec_v_ks_sctr.to_html(full_html=False, include_plotlyjs='cdn')
            
#########################
## Produce HTML Report ##
#########################           
page_title_text='Measuring Social Disparity in ML Models'
            
dp_hm_html = ''
for key in dp_hm_dict:
    dp_hm_html += str(dp_hm_dict[key])
eo_hm_html = ''
for key in eo_hm_dict:
    eo_hm_html += str(eo_hm_dict[key])
prec_hm_html = ''
for key in prec_hm_dict:
    prec_hm_html += str(prec_hm_dict[key])
ks_hm_html = ''
for key in ks_hm_dict:
    ks_hm_html += str(ks_hm_dict[key])
all_metrics_bp_html = ''
for key in all_metrics_bp_dict:
    all_metrics_bp_html += str(all_metrics_bp_dict[key])
# In[6]


# 2. Combine them together using a long f-string
html = f'''
    <!DOCTYPE html>
<html lang="en">
   <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <meta name="description" content="">
      <meta name="author" content="">
      <title>Team CVP</title>
      <!-- Bootstrap core CSS -->
      <link href="css/bootstrap.css" rel="stylesheet">
      <!-- Custom CSS for the '3 Col Portfolio' Template -->
      <link href="css/portfolio-item.css" rel="stylesheet">
   </head>
   <body>
      <nav class="navbar navbar-fixed-top navbar-inverse" role="navigation">
         <div class="container">
            <div class="navbar-header">
               <button type="button" class="navbar-toggle" data-toggle="collapse" data-target=".navbar-ex1-collapse">
               <span class="sr-only">Toggle navigation</span>
               <span class="icon-bar"></span>
               <span class="icon-bar"></span>
               <span class="icon-bar"></span>
               </button>
               <a class="navbar-brand" href="https://expeditionhacks.com/bias-detection-healthcare/" target="_blank">NCATS Bias Detection Challenge</a>
            </div>
            <!-- Collect the nav links, forms, and other content for toggling -->
            <div class="collapse navbar-collapse navbar-ex1-collapse">
               <ul class="nav navbar-nav">
                  <li><a href="#metrics">Define Metrics</a></li>
                  <li><a href="#identify-disparity">Identify Disparities</a></li>
                  <li><a href="#viz">Visualize</a></li>
                  <li><a href="#overall">Compare All</a></li>
               </ul>
            </div>
            <!-- /.navbar-collapse -->
         </div>
         <!-- /.container -->
      </nav>
      <div class="container">
      <div class="row">
         <div class="col-lg-12">
            <h1 class="page-header">Measuring Social Disparity in ML Models <small>by Team CVP</small></h1>
         </div>
      </div>
      <div class="row">
         <div class="col-md-6" id="metrics">
            <h3>Metrics</h3>
            <p>We use several different metrics to measure disparity. This includes:</p>
            <ul>
               <li>Demographic Parity</li>
               <li>Equalized Opportunity</li>
               <li>Precision</li>
               <li>KS (Kolmogorov Smirnov) Statistic</li>
            </ul>
            <h3>Chi-Squared Testing</h3>
            <p>Our model Chi-Squared Testing to Determine if Significant Disparity Exists</p>
            <p>If the value in the Reject column is True, there is some statistical difference in the metric rates of the Protected 
               Groups within the associated Protected Group Type. To prevent cases of only slightly different sets of rates 
               being rejected, a critical value of 0.01. Doing so reduces the probability that the differences are due to chance.
            </p>
         </div>
         <div class="col-md-6">
            <h3>Demographic Parity</h3>
            <p>A fairness metric that is satisfied if the results of a model's classification are not dependent on a given sensitive attribute</p>
            <p>
               {group_dp_chi}
            </p>
         </div>
         <div class="col-md-6">
            <h3>Equalized Opportunity</h3>
            <p>Do the same proportion of each population receive positive outcomes?</p>
            <p>
               {group_eo_chi}
            </p>
         </div>
         <div class="col-md-6">
            <h3>Precision</h3>
            <p>The quality of a positive prediction made by the model. Precision refers to the number of true positives divided by the total number of positive predictions (i.e., the number of true positives plus the number of false positives).</p>
            <p>
               {group_prec_chi}
            </p>
         </div>
         <div class="col-md-6">
            <h3>KS Statistic</h3>
            <p>Since the KS statistic is a difference in proportions (TPR-FPR), a chi-squared test cannot be performed. Instead, the 
               80% rule-of-thumb is used to ensure that, for a Protected Group Type, the minimum KS statistic is no less than 
               80% of the maximum KS statistic. If the value in the Meaningful Disparity column is true, there is an issue in
               Differential Validity for that Protected Group Type.
            </p>
            <p>
               {group_ks_80}
            </p>
         </div>
      </div>
      <div class="row" id="identify-disparity">
      <div class="col-lg-12" >
         <h3 class="page-header">Identify where the disparity lies</h3>
         <p>We use Paired Group Chi-Squared Testing to determine where the disparity lies. Tests to see which combinations of protected groups are significantly different for the specific metric.</p>
         <p>These heat maps show which combinations are statistically different. The higher the p-value, the lower the chance that
            the groups differ. The lower the p-value, the higher the chance. If the p-value is less than the critical
            value of 0.01, the corresponding square will be darker blue and be considered statistically different.
         </p>
      </div>
      <div class="col-md-6">
         <h3>Demographic Parity</h3>
         {dp_hm_html}
      </div>
      <div class="col-md-6">
         <h3>Equalized Opportunity</h3>
         {eo_hm_html}
      </div>
      <div class="col-md-6">
        <h3>Precision</h3>
        {prec_hm_html}
     </div>
     <div class="col-md-6">
        <h3>KS Statistic</h3>
        <p>To determine which combinations of protected groups are significantly different for the specific metric, we can use the 80% rule of thumb test. </p>
        <p>These heatmaps show which combinations are meaningfully different. The higher the proportion, the less disparity between
            the two respective groups. If the proportion's value is less than 0.8 (smaller value less than 80% of larger value)
            the two groups are deemed to be meaningfully different. </p>
        {ks_hm_html}
     </div>
      <div class="row" id="viz">
         <div class="col-lg-12">
            <h3 class="page-header">Visualizing the Metrics</h3>
            <p>All metrics within protected group types compared against each other</p>
            <p>Precision metric calculated by TP/(TP+FP) what proportion of the Demographic Parity score
               is due to True Positives. Suppose a protected group has much higher Demographic Parity scores than other groups but also higher False Observation rate. In that case, the objective should be to reduce the False Positive Rate and not boost it for the other groups.
            </p>
            <div class="col-md-12">
              {all_metrics_bp_html}
            </div>
         </div>
         <div class="col-lg-12">
            <h3 class="page-header">Comparing metrics for all Protected Groups against the number of observations</h3>
            <p>These scatter plots show whether the groups' metric values and the numbers of observations are correlated (linear relationship). 
                Compare only groups of the same type. Whether positive or negative, correlation implies that the metric values for the less frequent groups may be different simply because the model has not learned those groups well and that there might be a feature in the model capturing inherently other characteristics of those groups.
            </p>
            <div class="col-md-6">
               <h3>Demographic Parity</h3>
               {dp_sctr_html}
            </div>
            <div class="col-md-6">
               <h3>Equalized Opportunity</h3>
               {eo_sctr_html}
            </div>
            <div class="col-md-6">
               <h3>Precision</h3>
               {prec_sctr_html}
            </div>
            <div class="col-md-6">
                <h3>KS Statistic</h3>
                {ks_sctr_html}
            </div>
            <div class="col-md-6">
               <h3>Social Disparity Score</h3>
               <p>The Social Disparity score is calculated by first calculating the Social Parity score by taking the harmonic mean of the social fairness metrics. Then that score's value is subtracted for each group from the highest score of that group type. As such, the highest-value groups will have a Social Disparity Score of 0. Suppose a clear linear relationship exists for the group types in this scatter plot. In that case, it indicates that the model is likely overpredicting positive outcomes when there are fewer data in addition to the general correlation implications discussed for the above two plots.
               </p>
               {sds_sctr_html}
            </div>             
         </div>
         </div>
         <div class="row"  id="overall">
            <div class="col-lg-12">
               <h3 class="page-header">Comparing main metrics against each other</h3>
               <p>This plot helps us discover (if there is a negative trend) whether Equalized Opportunity is coming at the cost of equity (Demographic Parity) and vice versa. Also, it helps to visualize cases where lower-scoring groups in one metric also see a doubled disadvantage in the other. An apparent, positive-sloping linear relationship would indicate such. We generally expect them to be positively correlated because Demographic Parity incorporates the numerator and denominator of Equalized Opportunity in its numerator and denominator, respectively. But if the relationship is solid, the real-world harm to the lower-scoring groups would be exacerbated.
               </p>
            </div>
            <div class="col-md-1"></div>
            <div class="col-md-9">
              {dp_v_eo_sctr_html}
              {dp_v_prec_sctr_html}
            {dp_v_ks_sctr_html}
            {eo_v_prec_sctr_html}
            {eo_v_ks_sctr_html}
            {prec_v_ks_sctr_html}
            </div>
         </div>
      </div>
      <!-- /.container -->
      <div class="container">
         <hr>
         <footer>
            <div class="row">
               <div class="col-lg-12">
                  <p>Copyright &copy; CVP LLC 2023</p>
               </div>
            </div>
         </footer>
      </div>
      <!-- /.container -->
      <!-- JavaScript -->
      <script src="js/jquery-1.10.2.js"></script>
      <script src="js/bootstrap.js"></script>
   </body>
</html>
    '''

# 3. Write the html string as an HTML file
if op_sys == 'Windows':
    with open(os.path.join(out_directory_windows, 'measure_report.html'), 'w') as f:
        f.write(html)
elif op_sys == 'Linux' or op_sys == 'Darwin':
    with open(os.path.join(out_directory_linux_unix, 'measure_report.html'), 'w') as f:
        f.write(html)
else: print('Error: Unknown OS!')
            
            
