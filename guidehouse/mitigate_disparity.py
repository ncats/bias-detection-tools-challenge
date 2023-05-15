import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from fairlearn.postprocessing import ThresholdOptimizer
import joblib



# custom function to apply social bias mitigation technique :

def mitigate_social_bias(training_df,testing_df,sensitive_feature,target_variable,base_sklearn_model,parameters_of_skelarn_model,cv,fairness_constraint):

     """"

    A function to take training and testing sets, transform them, apply mitigation techniques, and generate dataframe of model's prediction to be used in the measure_disparity.py:
    training_df: the path to where the training dataset is stored
    testing_df:  the path to where the testing dataset is stored 
    sensitive_feature: the variable(s) we identify as sensitive/protected
    target_variable: the reponse variable in the dataset ( must be binay ) 
    base_sklearn_model: the original scikit-learn model the user wishes to use
    parameters_of_skelarn_model: the paramters associated with the base_sklearn_model to be used in GridSearchCV ( in a dictionary format )
    cv: the number of folds for cross validation when applying GridSearchCV
    fairness_constraint: the fairness metric to focus on when debiasing. We strongly recommend "equalized_odds" , but 'demographic_parity' or "true_positive_rate" i.e the equal opportinuty metric can be used. 

    """
     
     training_df=pd.read_csv(training_df)
     testing_df=pd.read_csv(testing_df)

     y=pd.concat([training_df[target_variable],testing_df[target_variable]])
     X=pd.concat([training_df.drop(target_variable,axis=1),testing_df.drop(target_variable,axis=1)])
     A=X[sensitive_feature]

     X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
     X, y, A, test_size=0.2, random_state=2023, stratify=y)

     X_train = X_train.reset_index(drop=True)
     X_test = X_test.reset_index(drop=True)
     y_train = y_train.reset_index(drop=True)
     y_test = y_test.reset_index(drop=True)
     A_train = A_train.reset_index(drop=True)
     A_test = A_test.reset_index(drop=True)

    
     numeric_features = X_train.select_dtypes('number').columns
     categorical_features = X_train.select_dtypes('object').columns
        

     numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"),
    MinMaxScaler())

     categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(sparse_output=False,dtype=int))

     col_transformer = make_column_transformer(
    (numeric_transformer, numeric_features), 
    (categorical_transformer, categorical_features),
    remainder='passthrough')
     

     model=GridSearchCV(base_sklearn_model, cv=cv , param_grid=parameters_of_skelarn_model)

     pipe = make_pipeline(col_transformer, model)

     pipe.fit(X_train, y_train)

     threshold_optimizer = ThresholdOptimizer(
    estimator=pipe,
    constraints=fairness_constraint,
    objective="balanced_accuracy_score",
    predict_method="predict",
    prefit=False,)

     threshold_optimizer.fit(X_train, y_train, sensitive_features=A_train)
     preds=threshold_optimizer.predict(X_test, sensitive_features=A_test)

     results=pd.DataFrame({'true_label':y_test.reset_index(drop=True),
                          sensitive_feature:X_test[sensitive_feature].reset_index(drop=True),
                          'predictions':pd.Series(preds).astype(int)})


     results.to_csv('inputs/post_debiasing_predictions.csv',index=False)

     file_name = 'outputs/Debiasing_model.sav'
     joblib.dump(threshold_optimizer, file_name)


mitigate_social_bias('inputs/training_df.csv','inputs/testing_df.csv','race_ethnicity','stroke',XGBClassifier(),{'n_estimators':[100,120]},5,'equalized_odds')
