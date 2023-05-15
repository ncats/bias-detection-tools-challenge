# iAdeptive AI/ML Bias Detection and Mitigation Tool ReadMe

## Implementation Instructions <br>

To run our code first of all notice how the code is separated into four python files, divided into two scripting files:
- measure_disparity.py
- mitigate_disparity.py

and two method files:
- measure_disparity_methods.py
- mitigate_disparity_methods.py

Make sure the two method files stay in a folder named "utils" located at the same location as the two script files. <br>
Then make sure that python version 3.9 or higher and the required modules are installed so that the tools can run with no problems. <br>

The required python dependencies for our tools are as follows:
- pandas 1.4.4
- plotnine 0.10.1
- scikit-learn 1.0.2

With all the required files and dependencies in place the user may begin to use this tool. <br>
For example when running from the command line, a settings ini file must first be set up, in a similar fashion to one of the three sample ini files provided. <br>
Once an appropriate ini file is made, the measure disparity tool could then be run like this:
```
python measure_disparity.py settings-measure.ini
```

Likewise the mitigate disparity tool could then be run like this:
```
python mitigate_disparity.py -p settings-model.ini
```

Both of these CLI commands will output tables in the terminal, as well as save graphs to a folder /graphs in your current workspace.

# Settings.ini Files
Three sample ini files are included, settings-measure.ini, settings-model.ini, and settings-agnostic.ini. Example usage is explained in the relevant sections. Below are descriptions for the various settings available in the INI file:

##### [dataset information]
- target_variable: name of target variable column
- input_variables: names of input variables column (only required for debiased_model object)
- demo_variables: names of demographic variables column
- advantaged_groups: list of advantaged group names
- probability_column: name of model probability column
- predicted_column: name of model predicted label column
- weight_column: name of weights column

*Example*:
```
target_variable: actual
input_variables: age,body_mass_index,body_weight,calcium,chloride,creatinine,urea_nitrogen,diabetes_flag
demo_variables: RACE,GENDER,INCOME
advantaged_groups: white,M,High
probability_column: proba
predicted_column: predicted
weight_column: weights
```

##### [model settings] (only required for debiased_model object)
- model_module: name of external module model to link
- model_class: name of external model class to link
- fit_method: name of external fit method for linked model
- predict_method: name of external predict method for linked model
- predict_proba_method: name of external predict probability method for linked model

*Example*:
```
model_module: xgboost
model_class: XGBClassifier
fit_method: fit
predict_method: predict
predict_proba_method: predict_proba
```

##### [model arguments] (only required for debiased_model object)
any model arguments you would like to pass on to the attached model_class in model settings

*Example*:
```
eta: 0.1
subsample: 0.7
max_depth: 8
scale_pos_weight: 5 
```

##### [paths]
- input_path: path for input data (csv)
- predict_input_path: path of data to predict (csv) (only required for debiased_model predictions)
- output_path: file for outputting data

*Example*:
```
input_path: ~/data/model_results.csv
predict_input_path: ~/data/data_to_predict.csv
output_path: debiased_predicted_dataset.csv
```


##### [other]
- display_metrics: list of metrics to display. See 'Available Bias Metrics' section
- optimization_repetitions: number of optimization repetitions. 20-30 recommended. Higher numbers will be more accurate but take a longer processing time. (only required for debiasing)
- debias_granularity: step size to test various thresholds. Lower numbers will be more accurate but take a longer processing time. Default is 0.01. (only required for debiasing)

*Example*:
```
display_metrics: STP,TPR,PPV,FPR,ACC
optimization_repetitions: 20
debias_granularity: 0.01
```

### Available Bias Metrics
The full names of the available metrics are listed below:

TP: True Positive,
TN: True Negative,
FP: False Postive,
FN: False Negative,
PPV: Positive Predictive Value,
TPR: True Positive Rate

| Abbreviation | Full Name | Calculation |
|---|---|---|
|STP| Statistical Parity|(TP + FP) / (TP + FP + TN + FN)|
|TPR| Equal Opportunity|TP / (TP + FN)|
|PPV| Predictive Parity|TP / (TP + FP)|
|FPR| Predictive Equality|FP / (FP + TN)|
|ACC| Accuracy|(TP + TN) / (TP + TN + FP + FN)|
|TNR| True Negative Rate|TN / (TN + FP)|
|NPV| Negative Predictive Value|TN / (TN + FN)|
|FNR| False Negative Rate|FN / (FN + TP)|
|FDR| False Discovery Rate|FP / (FP + TP)|
|FOR| False Omission Rate|FN / (FN + TN)|
|TS| Threat Score|TP / (TP + FN + FP)|
|FS| F1 Score |(2 \* PPV \* TPR) / (PPV + TPR)|


# Bias Measurement Tool
## CLI tool usage: 
measure_disparity.py [-h] settings-measure.ini <br>

Settings	:	settings.ini file <br>
-h, --help	:	show this help message and exit <br>

The settings ini file for must include [dataset information], [paths], and [other]. <br>

## Sample CLI usages: <br>
python measure_disparity.py settings-measure.ini <br>
python measure_disparity.py -h <br>

### Sample settings ini file: <br>
```
[dataset information]
target_variable: actual
demo_variables: RACE,GENDER,INCOME
advantaged_groups: white,M,High
probability_column: proba
predicted_column: predicted
weight_column: n/a

[paths]
input_path: ~/data/model_results.csv

[other]
display_metrics: STP,TPR,PPV,FPR,ACC
```
### measure_disparity_methods functions
Besides the CLI the tool can also be more directly used and customized by importing measure_disparity_methods. The main functions of measure_disparity_methods are described below. <br>


#### measure_disparity_methods.measured class
##### \_\_init\_\_ 
Function to intialize a measured object from measure_disparity_methods.py <br>

*Arguments* <br>
dataset <class 'pandas.core.frame.DataFrame'>: data generated from a model you which to analyze <br>
democols <class 'list'>: list of strings of demographic column names <br>
intercols <class 'list'>: list of lists with pairs of strings of demo column names you wish to see interactions between <br>
actualcol <class 'str'>: string of column name with actual values formatted as 0 or 1 <br>
predictedcol <class 'str'>: string of column name with predicted values formatted as 0 or 1 <br>
probabilitycol <class 'str'>: string of column name with probability values formatted as floats 0 to 1 <br>
weightscol <class 'str'>: string of column name with weights formatted as ints or floats <br>

*Returns* <br>
<class 'measure_disparity.measured'>: initialized measured object <br>

##### measured.MetricPlots
Function to draw a plot of any number of given metrics for a given demographic column <br>

*Arguments* <br>
colname <class 'str'>: string of demographic column name <br>
privileged <class 'str'>: string of name for the privileged subgroup within this demographic column <br>
draw <class 'bool'>: boolean of whether to draw the plot or not <br>
metrics <class 'list'>: list of strings of shorthand names of metrics to make graphs of <br>
graphpath <class 'str'>: string of folder path to where the plot should be saved as a png. If None then the plot will not be saved

*Returns* <br>
<class 'plotnine.ggplot.ggplot'>: plotnine plot of the metrics and demographics chosen <br>

##### measured.RocPlots
Function to draw two graphs of the Receiver Operating Characteristic curves for a demographic column <br>

*Arguments* <br>
colname <class 'str'>: string of demographic column name <br>
draw <class 'bool'>: boolean of whether to draw the graphs or not <br>
graphpath <class 'str'>: string of folder path to where the plot should be saved as a png. If None then the plot will not be saved <br>

*Returns* <br>
<class 'plotnine.ggplot.ggplot'>: plotnine graph of the ROC curve <br>
<class 'plotnine.ggplot.ggplot'>: plotnine graph of the ROC curve zoomed in on the upper left hand quadrant <br>

##### measured.PrintMetrics
Function to print out all chosen metrics for all chosen demographic columns in a table <br>

*Arguments* <br>
columnlist <class 'list'>: list of strings of the names of the demographic columns to print metrics for <br>
metrics <class 'list'>: list of shorthand names of the metrics to print out <br>

#### measured.PrintRatios
Function to calculate the ratio of metric values for one demographic column <br>
*Arguments* <br>
colname <class 'str'>: string of demographic column name <br>
privileged <class 'str'>: string of name for the privileged subgroup within this demographic column <br>
metrics <class 'list'>: list of strings of shorthand names of metrics to calculate ratios for <br>
printout <class 'bool'>: boolean of whether or not to print out the table of ratios calculated <br>

*Returns* <br>
metricsdf <class 'pandas.core.frame.DataFrame'>: table of the ratios calculated <br>

##### measured.fullnames
measure_disparity_methods.measured also has a class variable fullnames which is a dictionary wherein the shorthand names are the keys and the full length names of the metrics are the values <br>

# Bias Mitigation Tool <br>
### CLI tool usage: 
mitigate_disparity.py [-h] [-a] [-f] [-p] settings.ini  <br>

Settings	:	settings.ini file  <br>
-h, --help	:	show this help message and exit  <br>
-a, --adjust	:	Model Agnostic Adjustment (default: False)  <br>
-f, --fit	:	Fit Debiased Model Thresholds (default: False)  <br>
-p, -predict	:	Fit Debiased Model and Predict (default: False)  <br>

The settings ini file for -a option must include [dataset information], [paths], and [other]. The settings ini file for the options -f and -p must include [model settings] and [model arguments] in addition to [dataset information], [paths], and [other]. <br>

### Sample CLI usages: <br>
python mitigate_disparity.py -a settings-agnostic.ini <br>
python mitigate_disparity.py -f settings-model.ini <br>
python mitigate_disparity.py -p settings-model.ini <br>
python mitigate_disparity.py -h <br>

### Sample settings ini file:
```
[model settings]
model_module: xgboost
model_class: XGBClassifier
fit_method: fit
predict_method: predict
predict_proba_method: predict_proba

[model arguments]
eta: 0.1
subsample: 0.7
max_depth: 8
scale_pos_weight: 5 

[dataset information]
target_variable: esrd_flag
input_variables: age,body_mass_index,body_weight,calcium,chloride,creatinine,urea_nitrogen,diabetes_flag
demo_variables: RACE,GENDER,INCOME
advantaged_groups: white,M,High
probability_column: proba
predicted_column: predicted
weight_column: n/a

[paths]
input_path: ~/data/modeling_dataset.csv
predict_input_path: ~/data/modeling_dataset.csv
output_path: debiased_predicted_dataset.csv

[other]
optimization_repetitions: 10
debias_granularity: 0.01
display_metrics: STP,TPR,PPV,FPR,ACC
```

### mitigate_disparity_methods functions
Besides the CLI the tool can also be more directly used and customized by importing mitigate_disparity_methods. The main functions of mitigate_disparity_methods are described below. <br>

##### model_agnostic_adjustment
*Arguments* <br>
modeling_dataset <class 'pandas.core.frame.DataFrame'>: data from another model to be debiased <br>
y <class 'pandas.core.series.Series'>: series of actual values, either 0 or 1, from the input data <br>
proba <class 'pandas.core.series.Series'>: series of probabilities 0 to 1, from the input data <br>
D <class 'pandas.core.frame.DataFrame'>: dataframe of demographic values with one row for each observation <br>
demo_dict <class 'dict'>: dictionary with the names of the demographic columns as keys and the names of the privileged subgroups as values <br>
reps <class 'int'>: number of repetitions that this function will run through to create debiased predictions <br>

*Returns* <br>
output <class 'pandas.core.frame.DataFrame'>: data from another model but with new debiased predictions added under the column "debiased_prediction" <br>


#### mitigate_disparity_methods.debiased_model class
##### \_\_init\_\_ 
Function to intialize a debiased model object from mitgate_disparity_methods.py <br>

*Arguments* <br>
 config <class 'configparser.ConfigParser'>: object with all the configuration settings to initialize an object of the debiased_model class <br>

##### debiased_model.fit
*Arguments* <br>

X <class 'pandas.core.frame.DataFrame'>: data of the various input variables used to train the model <br>
y  <class 'pandas.core.series.Series'>: series of actual values from the modelling input data <br>
D <class 'pandas.core.frame.DataFrame'>: dataframe of demographic values with one row for each observation <br>
demo_dict <class 'dict'>: dictionary with the names of the demographic columns as keys and the names of the privileged subgroups as values <br>
reps <class 'int'>: number of repetitions the program will go through to create the new debiased cutoff <br>
thresholds args, kwargs: any number of arguments to be passed through from [model settings] and [model arguments] to whatever chosen module the users wishes to use for their machine learning model <br>

##### debiased_model.predict
*Arguments* <br>
X <class 'pandas.core.frame.DataFrame'>: data of the various input variables used to train the model <br>
D <class 'pandas.core.frame.DataFrame'>: dataframe of demographic values with one row for each observation <br>

*Returns* <br>
prediction_df <class 'pandas.core.frame.DataFrame'>: dataframe of just one column containing the new predicted values for each observation <br>

##### debiased_model.get_debiased_thresholds() <br>
*Returns* <br>
debiased_thresholds <class 'pandas.core.frame.DataFrame'>: new cutoff thresholds calculated to be debiased across all the demographic subgroups and interactions between subgroups <br>
