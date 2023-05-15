############################################
################ SETUP ######################
############################################
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
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from plotly import express as px
from math import asin, sqrt

# Define OS
try: 
    op_sys = platform.system()
except Exception as e:
    print('Error:', e)

# Define Output Model Paths for OS
outm_directory_linux_unix = r'../output_model'
outm_directory_windows =r'..\output_model'

# Define Output Report Paths for OS
out_directory_linux_unix = r'../reports'
out_directory_windows = r'..\reports'

###############
## FUNCTIONS ##
###############

# Ratios function used in imbalance_test()
def ratios_func(feature,df, y_bar):
    group_cnt=df[[y_bar,feature]].groupby(feature).count().reset_index().copy().rename(columns={str(y_bar):str('group_cnt')}).reset_index()
    gt_pos_cnt=df[[y_bar,feature]].groupby(feature).sum().reset_index().copy().rename(columns={y_bar:'pos_cnt'}).reset_index()
    imb_gt=pd.merge(group_cnt,gt_pos_cnt)
    imb_gt['neg_cnt']=imb_gt['group_cnt']-imb_gt['pos_cnt']
    imb_gt['group_type']=str(feature)
    imb_gt=imb_gt.rename(columns={str(feature):'group_name'})                    
    return imb_gt 

# Calculate whether certain groups are imbalanced
def imbalance_test(imb_gt):
    rule_of_thumb_num=.65
    rule_of_thumb_den=.35
    rule_of_thumb=rule_of_thumb_num/rule_of_thumb_den
    imb_gt['pos_ratio']=imb_gt['pos_cnt']/imb_gt['group_cnt']
    imb_gt['neg_ratio']=imb_gt['neg_cnt']/imb_gt['group_cnt']
    if any(imb_gt.loc[imb_gt['pos_ratio']>=0.5]):
        imb_gt.loc[imb_gt['pos_ratio']>rule_of_thumb_num,'balance_test']='imbalanced'
        imb_gt.loc[imb_gt['pos_ratio']<=rule_of_thumb_num,'balance_test']='balanced'
        imb_gt.loc[imb_gt['balance_test'].str.contains('imbalanced'),'Recommendation'] = \
            ' Investigate if any biases inherent to the data caused by clinical, administrative, data collection etc. practices'\
            ' particular to this group may be causing this imbalance. Worth checking for data quality issues.'
    elif any(imb_gt.loc[df['pos_ratio']<=0.5]):   
        imb_gt.loc[imb_gt['pos_ratio']<rule_of_thumb_den,'balance_test']='imbalanced'
        imb_gt.loc[imb_gt['pos_ratio']>=rule_of_thumb_den,'balance_test']='balanced'
        imb_gt.loc[imb_gt['balance_test'].str.contains('imbalanced'),'Recommendation'] = \
            ' Investigate if any biases inherent to the data caused by clinical, administrative, data collection etc. practices'\
            ' particular to this group may be causing this imbalance. Worth checking for data quality issues.'
    else:
        return
    if any(imb_gt.loc[imb_gt['neg_cnt']==0]): 
        imb_gt.loc[imb_gt['neg_cnt']==0, 'Critical Imbalance?']='Yes, find a way to collect more data for this group.'\
            ' If that is not possible, consider removing this group from the measurement and mitigation process. Predictions'\
            ' made especially in models where the corresponding protected feaure is included as a model feature will likely'\
            ' choose a positive value every single time for this group. The real proportion of positive outcomes may be far from 1.'
    elif any(imb_gt.loc[imb_gt['pos_cnt']==0]):
        imb_gt.loc[imb_gt['pos_cnt']==0, 'Critical Imbalance?']='Yes, find a way to collect more data for this group.'\
            ' If that is not possible, consider removing this group from the measurement and mitigation process. Predictions'\
            ' made especially in models where the corresponding protected feaure is included as a model feature will likely'\
            ' choose a negative value every single time for this group. The real proportion of negative outcomes may be far from 1.'
    else:
        return
    if any(imb_gt.loc[imb_gt['group_cnt']<40]):
        imb_gt.loc[imb_gt['group_cnt']<40, 'Adequate group sample size?']='No, this group has so few observations that it is likely'\
            ' to significantly skew the bias metrics, making it seem as if there is more disparity than there truly is.'\
            ' Strongly consider removing it from the measure and mitigation solutions.'
    else:
        return  
    if any(imb_gt.loc[imb_gt['neg_cnt']<40]): 
        imb_gt.loc[imb_gt['neg_cnt']<40, 'Adequate negative sample size?']='No, consider whether group has practical'\
            ' meaning or worth binning with another group. If not, consider dropping it from measurement and mitigation.'
    else:
        return
    if any(imb_gt.loc[imb_gt['pos_cnt']<40]):
        imb_gt.loc[imb_gt['pos_cnt']<40, 'Adequate positive sample size?']='No, consider whether group has practical'\
            ' meaning or worth binning with another group. If not, consider dropping it from measurement and mitigation.'
    else:
        return
    imb_gt['Recommendation']=imb_gt['Recommendation'].fillna('None')
    imb_gt['Critical Imbalance?']=imb_gt['Critical Imbalance?'].fillna('No')
    imb_gt['Adequate group sample size?']=imb_gt['Adequate group sample size?'].fillna('Yes')
    imb_gt['Adequate negative sample size?']=imb_gt['Adequate negative sample size?'].fillna('Yes')
    imb_gt['Adequate positive sample size?']=imb_gt['Adequate positive sample size?'].fillna('Yes')
    return imb_gt

# SMOTE + Tomek Links and SMOTE for data with categorical features + Tomek Links
def smote(df, features, y_bar):
    smt=SMOTE(random_state=42)
    smttomek = SMOTETomek(smote=smt, random_state=42)
    X, y = smttomek.fit_resample(df[features], df[y_bar])
    df1 = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    cols = features + [y_bar]
    df1.columns = cols
    return df1     

def smotenc(df, features, y_bar, cat_indices):
    smtnc = SMOTENC(sampling_strategy = 'minority', categorical_features=cat_indices, random_state = 42)
    tmk = TomekLinks(sampling_strategy = 'majority')
    smttomek = SMOTETomek(smote=smtnc, random_state=42)
    X, y = smttomek.fit_resample(df[features], df[y_bar])
    df1 = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    cols = features + [y_bar]
    df1.columns = cols
    return df1     

# LightGBM training and predicting
def lgbm_training(X1,Y1,SW,model_features,cat_features,params):
    df_train_lgb = lgb.Dataset(X1,label=Y1,weight=SW)
    lgb_booster = lgb.train(params,train_set=df_train_lgb, verbose_eval=True,
                            feature_name=model_features, categorical_feature=cat_features)
    return lgb_booster

def booster_predict(X1,model):
    lgbm_predict=model.predict(X1)
    return lgbm_predict

def MetricsReport(y_bar, y_hat, group_type, group_name, samp_weight):
    dict_metrics = {}
    dict_metrics['Protected Group Type'] = group_type        
    dict_metrics['Protected Group'] = group_name

            
    # Confusion Matrix
    tn, fp, fn, tp = metrics.confusion_matrix(y_bar,y_hat, labels=[0,1], sample_weight=samp_weight).ravel()
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

def MetricsReport_Thresh(y_bar, y_hat, group_type, group_name, samp_weight):
    dict_metrics = {}
    dict_metrics['Protected Group Type'] = group_type        
    dict_metrics['Protected Group'] = group_name
            
    # Confusion Matrix
    tn, fp, fn, tp = metrics.confusion_matrix(y_bar,y_hat, labels=[0,1], sample_weight=samp_weight).ravel()
    dict_metrics['TN'] = '{:.0f}'.format(tn)
    dict_metrics['FP'] = '{:.0f}'.format(fp)
    dict_metrics['FN'] = '{:.0f}'.format(fn)
    dict_metrics['TP'] = '{:.0f}'.format(tp)
            
    # Threshold Demographic Parity
    dem_par_t = (tp+fp)/(tp+fp+tn+fn+1)
    dict_metrics['Demographic Parity'] = '{:.5f}'.format(dem_par_t)   
           
    # Threshold Precision
    prec_t = (tp)/(tp+fp+1)
    dict_metrics['Precision'] = '{:.5f}'.format(prec_t)
            
    # Threshold Equalized Opportunity
    eq_op_t = (tp)/(tp+fn+1)
    dict_metrics['Equalized Opportunity'] = '{:.5f}'.format(eq_op_t)
    
    # Threshold False Positive Rate
    fpr_t = (fp)/(fp+tn+1)
    dict_metrics['FPR'] = '{:.5f}'.format(fpr_t)
    
    # Total Fairness
    total_fairness = statistics.harmonic_mean([dem_par_t, eq_op_t, prec_t, (1-fpr_t)])
    dict_metrics['Total Fairness Statistic'] = '{:.5f}'.format(total_fairness)
    
    return dict_metrics


############################
## CONSTRUCT CLASS OBJECT ##
############################

class Mitigator:
 

    def __init__(self, protected_features, model_features, cat_features, y_bar, samp_weight):
        self.protectedFeatures = protected_features
        self.modelFeatures = model_features
        self.catFeatures = cat_features
        self.yBar = y_bar
        self.sampWeight = samp_weight
        self.allFeatures = self.modelFeatures + list(set(self.protectedFeatures) - set(self.modelFeatures))
        self.catProt = self.catFeatures + list(set(self.protectedFeatures) - set(self.catFeatures))
        self.predProb = 'prediction_probability'
        self.lgbParams = {"objective": "binary", 
                          "metric": "binary_error",
                          "verbosity": -1,
                          "boosting_type": "gbdt",
                          "seed": 538,
                          "learning_rate": .1,
                          "num_leaves": 2
                         } 

 

    def transform(self, train_data):
        '''
        1. Tests train dataset imbalance per group per protected feature (produced after user's data cleaning and train test 
            split. It should contain all features include target y variable).
        2. Performs SMOTE with TomekLinks removal to rebalance data.
        3. Retests smoted train dataset for imbalance.
        4. Creates a before and after SMOTE report: imbalance_report.html in the reports folder
        5. Saves output predicted dataframe as CSV in output_model directory as rebalanced_train.csv
        '''
        # Imbalance Testing
        final_df=pd.DataFrame([])
        pre_imb_df=pd.DataFrame([])
        imb_df=pd.DataFrame([])
        final_imb_df=pd.DataFrame([])
        for feature in self.protectedFeatures:   
            feature_values = train_data[feature].unique().tolist()
            pre_imb=ratios_func(feature,train_data,self.yBar)
            pre_imb_df=pd.concat([pre_imb_df,pd.DataFrame(pre_imb)],
                                 ignore_index=True)
            imb_test=imbalance_test(pre_imb)
            imb_df=pd.concat([imb_df,pd.DataFrame(imb_test)], ignore_index=True)
        final_df=pd.concat([final_df,pre_imb_df],ignore_index=True) 
        final_imb_df=pd.concat([final_imb_df,imb_df],ignore_index=True)
        final_imb_df=final_imb_df[['group_type', 'group_name', 'group_cnt', 'pos_cnt', 'neg_cnt', 'pos_ratio', 'neg_ratio', 
                                   'balance_test', 'Recommendation', 'Critical Imbalance?', 'Adequate group sample size?', 
                                   'Adequate positive sample size?', 'Adequate negative sample size?']]
        # Convert to HTML
        before_imbalance = final_imb_df.to_html(classes='table table', index=False)       
        # Create field that concatenates protected features to expose unique combinations
        train_data[self.protectedFeatures] = train_data[self.protectedFeatures].astype(str)
        train_data['pf_combo'] = np.add.reduce(train_data[self.protectedFeatures], axis=1)
        # Run SMOTE-TomekLinks for each protected feature combination subset if there are enough observations
        num_obs = 1
        checkpoint = False
        while checkpoint is False:
            try:
                smoted_df = pd.DataFrame(columns=(self.allFeatures+[self.sampWeight]+[self.yBar]))
                for val in train_data.pf_combo.unique().tolist():
                    split = train_data.loc[train_data.pf_combo==val]
                    if len(split) < num_obs:
                        smt_df = split[self.allFeatures+[self.sampWeight]+[self.yBar]]
                    elif len(self.catProt) > 0:
                        cat_indices = [split.columns.get_loc(col) for col in self.catProt]
                        smt_df = smotenc(split, (self.allFeatures+[self.sampWeight]), self.yBar, cat_indices)
                    elif len(self.catProt) == 0:
                        smt_df = smote(split, (self.allFeatures+[self.sampWeight]), self.yBar)
                    smoted_df = pd.concat([smoted_df,smt_df], ignore_index=True)
            except ValueError:
                num_obs = num_obs + 10
                checkpoint = False
                print('smoting...')
                continue
            else:
                checkpoint = True
                print('train data smoting complete')
            break
        # Reformat columns
        smoted_df = smoted_df.convert_dtypes(infer_objects=True)
        d1 = dict.fromkeys(smoted_df.select_dtypes(np.int64).columns, int)
        d2 = dict.fromkeys(smoted_df.select_dtypes(np.float64).columns, np.float32)
        d3 = dict.fromkeys(smoted_df.select_dtypes(np.int32).columns, int)
        smoted_df = smoted_df.astype(d1)
        smoted_df = smoted_df.astype(d2)
        smoted_df = smoted_df.astype(d3)
        smoted_df[self.catProt] = smoted_df[self.catProt].astype(int)    
        # Retest for imbalances
        final_df=pd.DataFrame([])
        pre_imb_df=pd.DataFrame([])
        imb_df=pd.DataFrame([])
        final_imb_df=pd.DataFrame([])
        for feature in self.protectedFeatures:   
            feature_values = smoted_df[feature].unique().tolist()
            pre_imb=ratios_func(feature,smoted_df, self.yBar)
            pre_imb_df=pd.concat([pre_imb_df,pd.DataFrame(pre_imb)],
                                 ignore_index=True)
            imb_test=imbalance_test(pre_imb)
            imb_df=pd.concat([imb_df,pd.DataFrame(imb_test)], ignore_index=True)
        final_df=pd.concat([final_df,pre_imb_df],ignore_index=True) 
        final_imb_df=pd.concat([final_imb_df,imb_df],ignore_index=True)
        final_imb_df=final_imb_df[['group_type', 'group_name', 'group_cnt', 'pos_cnt', 'neg_cnt', 'pos_ratio', 'neg_ratio', 
                                   'balance_test', 'Recommendation', 'Critical Imbalance?', 'Adequate group sample size?', 
                                   'Adequate positive sample size?', 'Adequate negative sample size?']]
        after_imbalance = final_imb_df.to_html(classes='table table', index=False)  
        # Generate report
        html = f'''
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta name="description" content="Imbalance Report">
                <meta name="author" content="CVP LLC">
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
                        <span class="sr-only"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        </button>
                        <a class="navbar-brand" href="https://expeditionhacks.com/bias-detection-healthcare/" target="_blank">Imbalance Report</a>
                        </div>
                        <!-- Collect the nav links, forms, and other content for toggling -->
                        <div class="collapse navbar-collapse navbar-ex1-collapse">
                        <ul class="nav navbar-nav">
                            <li><a href="#before">Before SMOTE</a></li>
                            <li><a href="#after">After SMOTE</a></li>
                        </ul>
                        </div>
                        <!-- /.navbar-collapse -->
                    </div>
                    <!-- /.container -->
                </nav>
                <div class="container">
                    <div class="row" id="before">
                        <h1>Imbalance Report: Before and After SMOTE-TOMEKLinks</h1>
                        <h2>Before</h2>
                        {before_imbalance}
                        <div id="after">
                        <h2>After</h2>
                        {after_imbalance}
                        </div>
                    </div>
                </div>
            </body>
            </html>
            '''
        if op_sys == 'Windows':
            with open(os.path.join(out_directory_windows, 'imbalance_report.html'), 'w') as f:
                f.write(html)
        elif op_sys == 'Linux' or op_sys == 'Darwin':
            with open(os.path.join(out_directory_linux_unix, 'imbalance_report.html'), 'w') as f:
                f.write(html)
        else: print('Error: Unknown OS!')  
        if op_sys == 'Windows':
            smoted_df.to_csv(os.path.join(outm_directory_windows, 'rebalanced_train.csv'), index=False)
        elif op_sys == 'Linux' or op_sys == 'Darwin':
            smoted_df.to_csv(os.path.join(outm_directory_linux_unix, 'rebalanced_train.csv'), index=False)
        else: print('Error: Unknown OS!')    
        return smoted_df
        
        
    def fit(self, data, params = None):
        '''
        1. Fixed to 'smote_df' dataframe produced by .transform() method. If not, running .transform() method or simply wanting 
            to run a different dataframe through this method, reassign 'smote_df' value to a dataframe of your choice.
        2. Takes 'params' as input params for LightGBM model. Default params are defined as self.lgbParams. 
            Replace with your own dictionary for added customization.  
        3. Formats Data
        4. Creates model
        '''
        if params is None:
            light_gbm_params = self.lgbParams
        else:
            light_gbm_params = params
        # Prepare data formats
        input_df = data.convert_dtypes(infer_objects=True)
        d1 = dict.fromkeys(input_df.select_dtypes(np.int64).columns, int)
        d2 = dict.fromkeys(input_df.select_dtypes(np.float64).columns, np.float32)
        d3 = dict.fromkeys(input_df.select_dtypes(np.int32).columns, int)
        input_df = input_df.astype(d1)
        input_df = input_df.astype(d2)
        input_df = input_df.astype(d3)
        input_df[self.catProt] = input_df[self.catProt].astype('category')
        # Train LightGBM model
        return lgbm_training(input_df[self.modelFeatures], input_df[self.yBar],
                             input_df[self.sampWeight], self.modelFeatures,
                             self.catFeatures, light_gbm_params)
        
        
    def predict(self, model, test_dataframe, thresh_tuning = True):
        '''
        1. Can only be run on lgbm_output_model created using .fit() method.
        2. Input the test_dataframe with all features, including target y variable, from when initial train-test split performed.
        3. Performs prediction.
        4. Saves output predicted dataframe as CSV in output_model directory as predicted_test.csv
        5. Outputs predicted dataframe as 'input_df'.
        6. If thresh_tuning = True (default), threshold tuning procedure is performed.
        7. Final output is input_df with column 'y_hat_tuned' containing the prediction probabilities binned to binary outcomes based on
            tuned thresholds.
        8. Saves output threshold tuned predicted dataframe as CSV in output_model directory as thresh_tuned_test.csv
        '''
        test_df = test_dataframe
        # Prepare test dataframe for prediction
        test_df[self.catProt] = test_df[self.catProt].astype('category')
        test_df = test_df.convert_dtypes(infer_objects=True)
        d1 = dict.fromkeys(test_df.select_dtypes(np.int64).columns, int)
        d2 = dict.fromkeys(test_df.select_dtypes(np.float64).columns, np.float32)
        d3 = dict.fromkeys(test_df.select_dtypes(np.int32).columns, int)
        test_df = test_df.astype(d1)
        test_df = test_df.astype(d2)
        test_df = test_df.astype(d3)
        # Run model prediction
        lgbm_model_pred = model.predict(test_df[self.modelFeatures])
        # Add prediction probabilities back to the test dataframe
        test_df[self.predProb] = lgbm_model_pred.tolist()
        test_df[self.catProt] = test_df[self.catProt].astype(int)
        # Save model output 
        if op_sys == 'Windows':
            test_df.to_csv(os.path.join(outm_directory_windows, 'predicted_test.csv'), index=False)
        elif op_sys == 'Linux' or op_sys == 'Darwin':
            test_df.to_csv(os.path.join(outm_directory_linux_unix, 'predicted_test.csv'), index=False)
        else: print('Error: Unknown OS!')    
        # Reset name
        input_df = test_df
        #Condition procedure on whether threshold tuning is selected
        if thresh_tuning is False:
            # Use ROC curve to set optimal model prediction threshold and create y_hat for binary prediction value
            fpr_roc, tpr_roc, threshold_positive = metrics.roc_curve(input_df[self.yBar], input_df[self.predProb])
            full_roc = pd.DataFrame(zip(fpr_roc, tpr_roc, threshold_positive), columns = ['fpr','tpr','thrsh'])
            full_roc['rate_diff'] = full_roc['tpr'] - full_roc['fpr']
            threshold_positive = full_roc.loc[full_roc.rate_diff.idxmax(), 'thrsh']
            input_df.loc[input_df[self.predProb] > threshold_positive, 'y_hat_tuned']=int(1)
            input_df[['y_hat_tuned']] = input_df[['y_hat_tuned']].fillna(value=0)
        elif thresh_tuning is True:
            #Rebalance test data to find optimal thresholds for each group
            # Create a field that concatenates protected features for each observation
            input_df[self.protectedFeatures] = input_df[self.protectedFeatures].astype(str)
            input_df['pf_combo'] = np.add.reduce(input_df[self.protectedFeatures], axis=1)
            input_df[self.predProb] = input_df[self.predProb].astype(float)
            # Run SMOTE-TomekLinks for each combination subset if there are enough observations
            num_obs = 1
            checkpoint = False
            while checkpoint is False:
                try:
                    smoted_input_df = pd.DataFrame(columns=(self.allFeatures + [self.predProb] + [self.yBar] + [self.sampWeight]))
                    for val in input_df.pf_combo.unique().tolist():
                        split = input_df.loc[input_df.pf_combo==val]
                        if len(split) < num_obs:
                            smt_df = split[self.allFeatures + [self.predProb] + [self.yBar] + [self.sampWeight]]
                        elif len(self.catProt) > 0:
                            cat_indices = [split.columns.get_loc(col) for col in self.catProt]
                            smt_df = smotenc(split, (self.allFeatures + [self.predProb] + [self.sampWeight]), self.yBar, cat_indices)
                        elif len(self.catProt) == 0:
                            smt_df = smote(split, (self.allFeatures + [self.predProb] + [self.sampWeight]), self.yBar)
                        smoted_input_df = pd.concat([smoted_input_df,smt_df], ignore_index=True)
                except ValueError:
                    num_obs = num_obs + 10
                    checkpoint = False
                    print('threshold smoting...')
                    continue
                else:
                    checkpoint = True
                    print('test data smoting complete')
                break
            # Reformat dataframe
            smoted_input_df = smoted_input_df.dropna()
            smoted_input_df = smoted_input_df.convert_dtypes(infer_objects=True)
            d1 = dict.fromkeys(smoted_input_df.select_dtypes(np.int64).columns, int)
            d2 = dict.fromkeys(smoted_input_df.select_dtypes(np.float64).columns, np.float32)
            d3 = dict.fromkeys(smoted_input_df.select_dtypes(np.int32).columns, int)
            smoted_input_df = smoted_input_df.astype(d1)
            smoted_input_df = smoted_input_df.astype(d2)
            smoted_input_df = smoted_input_df.astype(d3)
            smoted_input_df[self.catProt] = smoted_input_df[self.catProt].astype(int)
            # Create dataframe to store metrics
            metrics_df = pd.DataFrame()
            thresh_space = np.linspace(0,1,101).tolist()
            # Calculate Optimal Thresholds
            for feature in self.protectedFeatures:   
                feature_values = smoted_input_df[feature].unique().tolist()
                for val in feature_values:
                    results_df = pd.DataFrame()
                    for t in thresh_space:
                        smoted_input_df.loc[smoted_input_df[self.predProb] > t, 'y_hat']=int(1)
                        smoted_input_df.loc[smoted_input_df[self.predProb] <= t, 'y_hat']=int(0)
                        smoted_input_df[['y_hat']] = smoted_input_df[['y_hat']].fillna(value=0)
                        result = MetricsReport_Thresh(list(zip(smoted_input_df[self.yBar].loc[smoted_input_df[feature]==val].tolist())),
                                                      list(zip(smoted_input_df.loc[smoted_input_df[feature]==val].y_hat.tolist())),
                                                      feature,
                                                      val,
                                                      smoted_input_df[self.sampWeight].loc[smoted_input_df[feature]==val])
                        result_df = pd.DataFrame([result])
                        result_df['threshold'] = t
                        result_df = result_df.fillna(0)
                        results_df = pd.concat([results_df,result_df], axis=0, ignore_index=True)
                        results_df['Total Fairness Statistic'] = results_df['Total Fairness Statistic'].astype('float64')
                        results_df = results_df.fillna(0)
                    max_results_df = results_df.sort_values(by=['Total Fairness Statistic','threshold'], ascending=False).head(1)
                    metrics_df = pd.concat([metrics_df,max_results_df], axis=0, ignore_index=True)
                    print('computing thresholds')
            # Pull out optimal threshold values
            metrics_df_sub = metrics_df[['Protected Group Type', 'Protected Group', 'threshold']]
            # Add each protected feature's threshold to data
            for feature in self.protectedFeatures:
                feature_values = smoted_input_df[str(feature)].unique().tolist()
                feature_dict = {}
                for k,v in metrics_df_sub.loc[metrics_df_sub['Protected Group Type']==str(feature)].groupby('Protected Group Type'):
                    for t in v.itertuples(index=False): 
                        temp_dict = {t[1]:t[2]}
                        feature_dict.update(temp_dict)
                smoted_input_df['threshold_' + str(feature)] = smoted_input_df[str(feature)].map(feature_dict)
            #Calculate averaged threshold and create y_hat_tune
            smoted_input_df['threshold'] = smoted_input_df.iloc[:,-len(self.protectedFeatures):].mean(axis=1)
            smoted_input_df.loc[smoted_input_df[self.predProb]>smoted_input_df['threshold'],'y_hat_tuned']=int(1)
            smoted_input_df=smoted_input_df.fillna(0).reset_index()
            # Save Model data into a dataframe
            if op_sys == 'Windows':
                smoted_input_df.to_csv(os.path.join(outm_directory_windows,'thresh_tuned_test.csv'))
            elif op_sys == 'Linux' or op_sys == 'Darwin':
                smoted_input_df.to_csv(os.path.join(outm_directory_linux_unix,'thresh_tuned_test.csv'))
            else: print('Error: Unknown OS!')
            return smoted_input_df
        else:
            print("Value of hreshold tuning parameter of .predict() method not valid. Please assign boolean or let default True apply")
            
            
    def measure(self, predict_df):
        '''
        1. Creates measure report of .predict() output
        '''
        input_df = predict_df
        # Calculate metrics for dataset-wide observations             
        total_model_results=MetricsReport(list(zip(input_df[self.yBar].tolist())),
                                          list(zip(input_df.y_hat_tuned.tolist())),
                                          'All Observations',
                                          'All Observations',
                                           input_df[self.sampWeight])
        # Create dataframe to store metrics
        metrics_df = pd.DataFrame([total_model_results])
        # Add metrics for each protected group within the protected features    
        for feature in self.protectedFeatures:   
            feature_values = input_df[feature].unique().tolist()
            for val in feature_values:
                results = MetricsReport(list(zip(input_df[self.yBar].loc[input_df[feature]==val].tolist())),
                                        list(zip(input_df.loc[input_df[feature]==val].y_hat_tuned.tolist())),
                                        feature,
                                        val,
                                        input_df[self.sampWeight].loc[input_df[feature]==val])
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
            .to_html(classes = 'table table', index=False)


        ######################            
        ## Chi-Squared Tests##
        ######################  

        # Group Demographic Parity 
        group_dp_chi = pd.DataFrame(columns=['Protected Group Type', 'Chi-Squared', 'P-Value'])       
        for cat in self.protectedFeatures:
            cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
            (chi2, p, arr) = ssp.proportions_chisquare(count=(cat_df.TP + cat_df.FP), nobs=(cat_df['Observation Count'])) 
            result = [cat, chi2, p]
            group_dp_chi.loc[len(group_dp_chi)] = result
        group_dp_chi['Fail'] = group_dp_chi['P-Value']<0.01
        group_dp_chi = group_dp_chi.to_html(classes='table table', index=False)


        # Group Equalized Opportunity           
        group_eo_chi = pd.DataFrame(columns=['Protected Group Type', 'Chi-Squared', 'P-Value'])       
        for cat in self.protectedFeatures:
            cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
            (chi2, p, arr) = ssp.proportions_chisquare(count=cat_df.TP, nobs=(cat_df.TP + cat_df.FN)) 
            result = [cat, chi2, p]
            group_eo_chi.loc[len(group_eo_chi)] = result
        group_eo_chi['Fail'] = group_eo_chi['P-Value']<0.01
        group_eo_chi = group_eo_chi.to_html(classes='table table', index=False)     

        # Group Precision Parity 
        group_prec_chi = pd.DataFrame(columns=['Protected Group Type', 'Chi-Squared', 'P-Value'])       
        for cat in self.protectedFeatures:
            cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
            (chi2, p, arr) = ssp.proportions_chisquare(count=(cat_df.TP), nobs=(cat_df.TP + cat_df.FP)) 
            result = [cat, chi2, p]
            group_prec_chi.loc[len(group_prec_chi)] = result
        group_prec_chi['Fail'] = group_prec_chi['P-Value']<0.01
        group_prec_chi = group_prec_chi.to_html(classes='table table', index=False)

        # Group KS Statistic Proportions 80% Rule of Thumb (Not Chi-Square)
        group_ks_80 = pd.DataFrame(columns=['Protected Group Type','KS Proportion'])
        for cat in self.protectedFeatures:
            cat_df = group_metrics.loc[group_metrics['Protected Group Type'] == cat]
            prop = cat_df['KS Statistic'].loc[cat_df['KS Statistic'].idxmin()]/cat_df['KS Statistic'].loc[cat_df['KS Statistic'].idxmax()]
            result = [cat, prop]
            group_ks_80.loc[len(group_ks_80)] = result
        group_ks_80['Meaningful Disparity'] = group_ks_80['KS Proportion']<0.8
        group_ks_80 = group_ks_80.to_html(classes='table table', index=False)

        # Paired Group Chi-Squared Test for Demographic Parity
        dp_hm_dict = {}
        for cat in self.protectedFeatures:
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
        for cat in self.protectedFeatures:
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
        for cat in self.protectedFeatures:
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
        for cat in self.protectedFeatures:
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
        for cat in self.protectedFeatures:
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
        page_title_text='Measuring Social Disparity in Mitigated Model'

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
                <p>To determine which combinations of protected groups are significantly different for the specific metric, we can use a 80% rule of thumb test. </p>
                <p>These heatmaps show which combinations are meaningfully different. The higher the proportion, the less disparity between
                    the two respective groups. If the proportion's value is less than 0.8 (smaller value less than 80% of larger value)
                    the two groups are deemed to be meaningfully different. </p>
                {ks_hm_html}
             </div>
              <div class="row" id="viz">
                 <div class="col-lg-12">
                    <h3 class="page-header">Visualizing the Metrics</h3>
                    <p>All metrics within protected group types compared against each other</p>
                    <p>False Observation metric calculated by FP/(TP+FP) what proportion of the Demographic Parity score
                       is due to False Positives. Suppose a group has much higher Demographic Parity scores than other groups but also higher False Observation rate. In that case, the first step should be to reduce the False Positive Rate and not boost it for the other groups.
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
            with open(os.path.join(out_directory_windows, 'mitigate_report.html'), 'w') as f:
                f.write(html)
        elif op_sys == 'Linux' or op_sys == 'Darwin':
            with open(os.path.join(out_directory_linux_unix, 'mitigate_report.html'), 'w') as f:
                f.write(html)
        else: print('Error: Unknown OS!')    
            
        return group_metrics
        
    

        
    