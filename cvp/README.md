
<!-- ABOUT THE PROJECT -->
## About This Development
This is Team CVP's solution to [NIH NCAT's Bias Detection Tools in HealthCare Challenge]( https://expeditionhacks.com/bias-detection-healthcare/)

### Overview
Artificial intelligence algorithms are increasingly being adopted as decision-making aids with the promise of overcoming biases of human 
decision-makers. Machine learning models used in this fashion may unintentionally amplify or even create bias because of choices made during
development, or they may become biased from data that they were trained on. Due to the increasing use of AI systems to supplement regular 
decision-making and deep-rooted disparities in the US healthcare system where this training data comes from, there have been growing demands 
for model transparency, explainability, and interpretability to determine the presence of bias. **CVP’s Data Science Team** investigated ways to 
automatically measure certain types of bias and mitigate them without human intervention. 

This tool was inspired by our hands-on experience that bias can manifest in many obvious and not-obvious forms. We devised a solution which 
aims to support **complex decision-making** by *giving people simple insights to make smart determinations*. This also helps us identify our own 
**inherent bias**, like confirmation  bias where we find what we expect. Bias is inevitable, but with effective tools, unwarranted bias (i.e., 
bias not inherent in the real world) can be minimized, and real-world bias can be better understood. <br />
*Please reference the **[Full Report](https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io/blob/main/Team%20CVP%20Submission%20-%20final.pdf)** for more information*

### Bias Detection Tool Architecture
![AI Bias Tool Architecture](https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io/blob/e32af3038ab9d87ac031e5171713adff2145dd73/assets/img/23-003-001-02_NCATS%20AI%20Challenge-02.png)

Our team successfully reduced social bias when training and optimizing a LightGBM (gradient-boosted decision tree) model through effective bias 
measurement and mitigation. Our solution is runnable on any modern PC, in an office or in the cloud, and can be used on any structured dataset. With only
one executable Python script and one Python module, it will first measure and then mitigate many types of bias, leading to more equitable healthcare 
outcomes across the country.

By following the installation directions below, the tool’s **architectural capabilities** include the following:  </br>
*	Run locally on a laptop or desktop  </br>
*	Run on physical or virtual server in the cloud </br>
*	Be loaded into a service like AWS Batch (Docker container) or AWS Lambda (Firecracker VM) which can run thousands of Python programs in
parallel. Datasets could be passed in via AWS API Gateway and reports delivered as HTML hosted S3 or CloudFront.

**Cloud Deployment** </br>
If deployed in a cloud server such as AWS or Azure, this solution can supplement a larger setup that harnesses existing AWS and Azure AI bias </br>
fairness and implementation tools, dashboards, and scorecards and other tools capable of reviewing many different models to provide output back </br>
to reviewers. </br>

### Bias Detection and Mitigation Tool for Clinical Decision Making
Our AI tool (Reference Architecture diagram above) aims to increase awareness of potential bias and facilitate stakeholder engagement and oversight by 
producing an automatically generated Measure Report on several measures like demographic parity and equalized opportunity. Instead of using 
predefined protected and reference classes, we analyze across entire **demographic or protected features**. We believe that the groups being 
discriminated against can change over time, and we do not want to introduce any bias by only examining certain classes. By examining all groups,
we are able to track these changes and assess holistic disparity. The report dives deep into each protected feature (e.g., race, age, gender) 
to show where bias is detected. This allows a knowledgeable reviewer, well-informed on the topic, to quickly spot where the bias is and decide 
on a course of action.

Our goal for this project was to develop a broad, user-friendly set of diagnostics and tools for the measurement and mitigation of bias, 
not a one size fits all approach that may worsen the problem of automation bias. The more we investigated the various forms of bias in AI, 
the more we realized they are several loosely related problems all falling under one umbrella, with no single indicator capable of summarizing 
bias. Much like the complexity surrounding “solving” the nation’s deficit, the identification, measurement, and mitigation of AI Bias requires 
a variety of information coupled with expert judgment to recognize sustainable improvements over time. 

### Mitigation Capabilities
* Calculating the Mitigation-centric Bias Measurements
* Rebalancing Transformation on Train SMOTE- TomekLinks
* Pre- and Post-Hoc Imbalance Reporting
* Population Parameter: Sample Weights
* Post Processing Evaluation: Threshold Tuning <br />

### Bias Metrics
![Metrics](https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io/blob/main/assets/img/Metrics_NIH_Challenge.png)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
Watch CVP's **[Bias Detection Tool Youtube Video](https://youtu.be/O4GG6Ph55U8)** to view a walkthrough the tool's usage.
<br /> 
* Below are instructions on setting up your project locally.
* To get a local copy up and running follow these simple example steps

*Note: We have also created a Google Colab notebook that contains the retrieval of the source code, setup of the environment, and running of our solution with example data. It is available at [THIS GOOGLE COLAB LINK](https://colab.research.google.com/drive/1KP64rF6k-DK5F83OYkTwWidvuVzPKhmo?usp=sharing)*

### Prerequisites
* Python 3.8
* [Requirments](https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io/blob/main/scripts/requirements.txt) are in `scripts\requirements.txt`
* Full report is in [Bias Detection Report- NCATS Challenge Submission](https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io/blob/main/Team%20CVP%20Submission%20-%20final.pdf).

### Folder Structure
The script is using the following directory tree structure:
```/
├── scripts/           # This is where <code>measure_disparity.py</code> and <code>mitigate_disparity.py</code> are
├── reports/           # Location for the html report generated by the script
├── input_model/       # Insert input model data here
├── output_model/      # This is where the mitigated model prediction is saved
├── data/              # Sample data to run the model 
├── js/                # javascript for the team submission landing page 
├── css/               # css files for the team submission landing page 
├── assets/            # images for the team submission landing page 
```
### Installation
The only technical requirements before running are a **Python environment setup on the computer** and a **Git client to retrieve the code**, 
though you could download our code from GitHub’s website if the latter is problematic. We recommend [Anaconda](https://www.anaconda.com/) which has setup directions here.    </br>
Note: CPU, RAM, and storage requirements are dependent on the size of the dataset you wish to measure and mitigate bias on. You should generally have
RAM that is greater than or equal to the dataset. <br>
<br>
**Once Python is available and an environment is activated:** <br />
<ol>
  <li> Run <code>Git clone</code> on the [cvp-bias-detection-healthcare.github.io](https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io) repository
    </li>
  <li> Run <code>pip install requirements.txt</code> </li>
 </ol>

#### Script Usage <br />
**To use the [measure_disparity.py](https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io/blob/main/scripts/measure_disparity.py) file:** <br />
<ol>
  <li>	Open “measure_disparity.py”, located in the scripts folder, in a text editor. </li> 
  <li> The first 15 lines of code are used for storing metadata about your dataset of interest. Please modify them as needed to locate the data, tag the protected <br /> features, and label which columns have the probabilities, the true labels and sample weights. Save your changes. </li> 
  <li> Run “python measure_disparity.py”. An HTML report (“measure_rerport.html”) will be automatically generated in the “reports” folder </li> 
  <li> Double click it or transfer to a computer with an internet connected web browser to review it. </li>
</ol>
 <br />
 
**To use the [mitigate_disparity.py](https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io/blob/main/scripts/mitigate_disparity.py) file:** <br />
<ol>
  <li>	As required by the challenge, the mitigate script returns a Python object/class with fit(), transform(), and predict() methods in addition to a measure() method for report generation. Read the README inline comments for the class creation instance and class methods to understand the arguments and parameters they need as input  </li>
  <li>	The mitigate_disparity.py file needs to be imported as a module in another notebook or a .py script. </li>
  <li>	For ease of use, we have included a “run_mitigate.ipynb” sample notebook which is setup to use the object and its methods to perform a full mitigation </li>
  <li>	You may run this notebook as is with our sample data file (“diabetes_data”) or copy its contents for use with your own dataset </li>
  <li>	For the latter, you will need to define your train and test pandas dataframes </li>
  <li>	We recommend predefining all of the critical fields (like in the measure script) that need be passed into the class and methods </li>
  <li>	For this script, the LightGBM model parameters dictionary is another critical field, and you can either keep it as is or modify as you see fit for your dataset </li>
  <li>	Once all required parameter inputs are defined, initialize the Mitigator object and call its methods in this order: transform(), fit(), predict(), measure(). These together will generate the “imbalance_report.html” and the “mitigate_report.html” (same format analysis as “measure_report.html”) in the reports folder. It also outputs the transformed train dataset, the predicted test dataset, and the threshold tuned predicted test dataset to help build an audit trail.</li>
</ol>

*For more detailed information on how the mitigation process works including what parameters are required, please reference the readme.txt file.*

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Clinical Setting Usage
*Note: This challenge and our tool assume you already have a trained LightGBM model from which you have used to generate predictions
for the test set. If the model is not LightGBM, steps 1-5 below are still functional*

To use our tool to evaluate and improve an ML model used in the prediction of an adverse event or diagnosis of a disease 
or condition, one should:
1.	Save the predicted test set as CSV.
2.	Run the measurement program which will produce a report showing how the model performed across and within the protected 
features.
3.	At a minimum, create a review group of a data scientist (ideally the one who made the model) and a medical practitioner who
is familiar with treating or diagnosing the condition. 
4.	Have this group examine the report to determine what was found to be biased and where they think the bias is coming from. 
The report will highlight areas of concern, but no AI will have visibility into sampling problems that may have impacted the 
training data or domain knowledge as to whether that disparity is normal. For example, if the model was for the diagnosis of 
sickle cell anemia, Alzheimer’s, Parkinson’s or another condition that has uneven impact across protected features (e.g., age,
race), this may be expected and not a large concern. 
5.	The review team should decide if the bias report has found apparent social/predictive biases or whether the issue may be 
systemic, sampling, or another type of bias where a new training dataset would be more appropriate.
6.	If team determines the issue is due to model bias and the model algorithm is not LightGBM, proceed to run a LightGBM version 
of the model. Rerun measure for it the mitigation tool. This will attempt to lower the apparent rate of disparity in the model and 
yield a fairer outcome.  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Ethical and Trustworthy AI 
* You don’t want unethical bias in the real world
* You do want bias in the data, because you want bias to reflect the real world
* You do want best fit bias in the data, because you want the result to accurately reflect the bias in the data
* You don’t want unethical bias in your business, so it is critical that you appropriately act upon accurate results from AI algorithms

![AI Bias](https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io/blob/main/assets/img/AI%20Bias.PNG)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

**Social Bias**
* **Protected Groups Definition:** "Prohibited grounds of discrimination" (French: motif de distinction illicite) in employment and housing are listed in the federal Canadian Human Rights Act as well as the provincial human rights codes.

* For example, the federal law lists: race, national or ethnic origin, colour, religion, age, sex, sexual orientation, gender identity or expression, marital status, family status, genetic characteristics, disability, and conviction for an offence for which a pardon has been granted or in respect of which a record suspension has been ordered.

**Considering Fairness: Unwarranted Bias**
* Maybe we find out that we have a very accurate model, but it still produces disparate outcomes. This may be unfortunate, but is it fair? 
* An alternative to accounting for unequal outcomes may be to look at the equality of error rates, and whether there are more mistakes for one group of people than another.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### AI Code of Conduct
<ol> 
<li> Must be able to disclose ambiguities of a tool being utilized and the risks that are taken</li>
<li> Must consider ethics across dynamics and systems </li>
<li> AI system operations must be able to justify organizational processes around models utilized and track data lineage </li>
<li> Operations and maturity stages should prioritize stakeholder perception and safety, accountability, transparency, risk assessment, and stakeholder engagement over implementation incentives, validating decisions utilizing complete documentation and written agreements</li>
<li> Technical documentation is required for all data handling, transformation, and insights to ensure transparency of records, quality, and complexity. </li>
<li> Risk, quality, safety and bias assessments should be performed periodically on every component, interface, and environment connected to the AI system across its lifecycle </li>
</ol>
 
 <p align="right">(<a href="#readme-top">back to top</a>)</p>
 
<!-- LICENSE -->
## License
Distributed under the BSD 3 License. See `LICENSE.txt` for more information.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact
* Manpreet Khural - manpreetkhural@cvpcorp.com
* Cal Zemelman - calzemelman@cvpcorp.com 
* Lauren Winstead - laurenwinstead@cvpcorp.com 
* Wei Chien - weichien@cvpcorp.com 
* Rose Anderson - roseanderson@cvpcorp.com
Project Link: [https://github.com/cvp-bias-detection-healthcare/](https://github.com/cvp-bias-detection-healthcare/cvp-bias-detection-healthcare.github.io/)

### NCATS POC
Have feedback or questions about this challenge? Please send your feedback or question to [NCATSAIBiasChallenge@mail.nih.gov](NCATSAIBiasChallenge@mail.nih.gov)
Alternate NCATS Challenge POC: [expeditionhacks@blueclarity.io](expeditionhacks@blueclarity.io)

### Supporting Links
* [NIST AI Framework](https://www.nist.gov/itl/ai-risk-management-framework)
<p align="right">(<a href="#readme-top">back to top</a>)</p>
