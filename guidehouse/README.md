# Bias_Detection_Tools

## Link to view all submission components

https://github.com/Saleh-Aldrees/Bias_Detection_Correction_Tools/blob/main/index.html

## Description: 

This repository is meant to address social bias issues in machine learning models in a healthcare setting. It targets binary classification problems by measuring social biases against certain demographic sub-groups in the predictions the model makes , mitigating those biases by focusing on improving the equalized odds metric , then measuring the social biases again to check the improvement in fairness metrics. 

The repository is general enough to be used in many applications as long as the datatsets are in the correct format. 

## Python Scripts:

The repository contains two pythin scripts , namely measure_disparaity.py and mitigate_disparaity.py

### measure_disparaity.py 

Contains a single function that takes a csv file that contains the original model's predictions, ground truth labels, and the sensitive feature column and generates a csv file and a graph depicting the fairness metric scores as well as performance scores by demographic subgroup in the output folder. 

The user just needs to specify the correct arguments for the fucntion as shown on the script and demonstrated in the video. 

### mitigate_disparaity.py

Contains a single function that takes training and testing csv files with specifying some other arguments , transforms the data and builds models , then generates a csv file that contains the debiasing model's predictions, ground truth labels, and the sensitive feature column in the input folder to be used to re-measure the bias using the measure_disparaity.py. The script also saves the debiasing model in the output folder.

The user just needs to specify the correct arguments for the fucntion as shown on the script and demonstrated in the video. 

## Installation

The user just needs to run the command **"pip install -r requirements.txt"** in the terminal to install all the packages required. 
