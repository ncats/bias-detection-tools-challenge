Once Python is available on your PC and an environment is activated:

1.	Run “Git clone https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io”
2.	Run “pip install requirements.txt”

To use the measure_disparity.py file:
1.	Open “measure_disparity.py”, located in the scripts folder, in a text editor.
2.	The first 15 lines of code are used for storing metadata about your dataset of interest. Please modify them as needed to locate the data, tag the protected features, and label which columns have the probabilities, the true labels and sample weights. Save your changes.
3.	Run “python measure_disparity.py”. An HTML report (“measure_rerport.html”) will be automatically generated in the “reports” folder
4.	Double click it or transfer to a computer with an internet connected web browser to review it.

To use the mitigate_disparity.py file:
1.	As required by the challenge, the mitigate script returns a Python object/class with fit(), transform(), and predict() methods in addition to a measure() method for report generation. Read the README inline comments for the class creation instance and class methods to understand the arguments and parameters they need as input
2.	The “mitigate_disparity.py” file needs to be imported as a module in another notebook or a .py script.
3.	For ease of use, we have included a “run_mitigate.ipynb” sample notebook which is setup to use the object and its methods to perform a full mitigation
4.	You may run this notebook as is with our sample data file (“diabetes_data”) or copy its contents for use with your own dataset
5.	For the latter, you will need to define your train and test pandas dataframes 
6.	We recommend predefining all of the critical fields (like in the measure script) that need be passed into the class and methods. An example of this can be seen in the run_mititgate.ipynb and Google colab notebooks as well. 
7.	!!!!MAKE SURE ALL MODEL FEATURES AND PROTECTED FEATURE COLUMNS HAVE FLOAT OR INT DATATYPES!!!! These are required for transform(), fit(), and predict() to run.
8.	For this script, the LightGBM model parameters dictionary is another critical field, and you can either keep it as is or modify as you see fit for your dataset
9.	Once all required parameter inputs are defined, initialize the Mitigator object and call its methods in this order: transform(), fit(), predict(), measure(). These together will generate the “imbalance_report.html” and the “mitigate_report.html” (same format analysis as “measure_report.html”) in the reports folder. It also outputs the transformed train dataset, the predicted test dataset, and the threshold tuned predicted test dataset to help build an audit trail. 

Make sure you have the right inputs for each mitigate_disparity.py class object and methods:
1.	Initialize class instance with .Mitigator(params)
    INPUT PARAMS IN ORDER: 
                          List of protected feature column names; Data type must be int
                          List of all model predictor/feature column names; Data type can be either int or float
                          List of all categorical feature column names; Data type must be int
                          String of true label/y_bar column name; Data type must be int
                          String of sample weights column name; Data type must be float
    OUTPUTS:
            In kernel: Class instance object
                 
2.	.transform(params) 
    INPUT PARAMS:
                 Name of pandas dataframe of the train dataset
    OUTPUT:
           In kernel: Pandas dataframe of rebalanced train dataset
           Exported: mbalance_report.html in reports folder
           Exported: rebalanced_train.csv in output_model folder containing the rebalanced train dataset
                 
3.	.fit()
    INPUT PARAMS IN ORDER:
                          Name of pandas dataframe of rebalanced train dataset
                          Dictionary containing LightGBM model parameters
    OUTPUT:
           In kernel: LightGBM model object
    
4.	.predict()
    INPUT PARAMS IN ORDER:
                          LIGHTGBM model object
                          Name of pandas dataframe of the test dataset
                          Boolean for Threshold Tuning (Default is True which means Threshold Tuning is performed)
    OUTPUT:
           In kernel: Pandas dataframe containing test data with model 'prediction_probabilities' and 'y_hat_tuned' columns
           Exported: predicted_test.csv in output_model folder containing the predicted but not threshold tuned test dataset
           Exported: thresh_tuned_test.csv in output_model folder containing post-threshold tuning test dataset
           
4.	.measure()
    INPUT PARAMS IN ORDER:
                          Name of pandas dataframe of the post threshold tuning test dataset
    OUTPUT:
           In kernel: Pandas dataframe containing scores for all metrics for all groups in all protected features
         

                
                 

