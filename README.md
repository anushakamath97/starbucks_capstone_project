# Starbucks Capstone Project

Starbucks sends various offers to its customers on a weekly basis to improve their engagement and experience with their products. Different customers respond differently with each offer. The customer behaviour also changes based on the time as they might have different needs at different situations.

Through this project we use the simulated customer transactions data to explore behaviour patterns and also try to predict how a customer would respond to different offers.


### Dataset Info

There are 3 datasets, below mentioned are the meaning of each column in each dataset.

**portfolio**

Containing offer ids and meta data about each offer.

- id (string) - offer id
- offer_type (string) - type of offer ie BOGO, discount, informational
- difficulty (int) - minimum required spend to complete an offer
- reward (int) - reward given for completing an offer
- duration (int) - time for offer to be open, in days
- channels (list of strings) - channels used to send offer

**profile**

Demographic data for each customer.

- age (int) - age of the customer 
- became_member_on (int) - date when customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer id
- income (float) - customer's income

**transcript**

Records for transactions, offers received, offers viewed, and offers completed.

- event (str) - record description (ie transaction, offer received, offer viewed, etc.)
- person (str) - customer id
- time (int) - time in hours since start of test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record

### Installations

Your python environment would need the following packages installed. Specifying the versions used while working on the projects.

+ `Flask==1.1.2`
+ `imbalanced-learn==0.8.0`
+ `matplotlib==3.4.2`
+ `numpy==1.20.2`
+ `notebook==6.3.0`
+ `pandas==1.2.4`
+ `plotly==4.14.3`
+ `requests==2.25.1`
+ `scikit-learn==0.24.1`

### Instructions:

1. Run all jupyter notebooks in following order:

    - Data_Exploration_Cleaning
    - Data_Visualization
    - Data_Preprocessing
    - Data_Modelling_Prediction

_NOTE: Training each model is commented in Data_Modelling_Prediction notebook and pre-trained models are loaded. Please uncomment for training again._


2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Structure and details

The files in this repository have the following structure:

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
| - data
| |- completion.csv # average completion time calculated for web app
| |- profile.csv # user data required for web app
| |- portfolio.csv   # offer data required for web app
|- run.py  # Flask file that runs app

- data
| - portfolio.csv  # offers data after cleaning 
| - profile.csv  # users data after cleaning
| - transcript.csv # events data after cleaning
| - portfolio.json  # data to process
| - profile.json  # data to process
| - transcript.json # data to process
| - processed
| |- data.csv # feature data for models
| |- target.csv # target data for models

- models
|- logistic_regression_1.pkl # trained on unbalanced data
|- logistic_regression.pkl # trained on SMOTE data
|- decision_tree.pkl # trained on SMOTE data
|- naive_bayes.pkl # trained on SMOTE data
|- gradient_boosting.pkl # # trained on SMOTE data
|- knn.pkl # trained on SMOTE data
|- decision_tree_2.pkl # trained on SMOTE & RandomUnderSampler data

- README.md
```

### Analysis

We have analysed different patterns in data like members joined per year, distribution of channel usage for offers, etc.

There is much data where the user has viewed and completed the offer than to where they have completed the offer by chance. This caused an unbalanced dataset for model training.

Customer age distribution is nearly normal which is a good factor as we did not have to account for any bias.
The data is consistent in terms if customers completing offers only of they had received them.

We saw that Decision Tree classifier performed best in this scenario compared to other supervised algorithms. The causes and reasoning is discussed in the detail in notebook. We have evaluated models with different metrics perspectives like precision, recall, accuracy. Confusion Matrices are plotted to visualize the performance.

### Conclusion

In this project we saw that Decision Tree, Gradient Boosting and KNN algorithms performed well than Naive Bayes and Logistic Regression.

An effort to balance the data in done using SMOTE and RandomUnderSampler, which helped improve the performance of models. 
SMOTE with RandomUnderSampler did not improve the peformance significantly and this could mean the models are unable to learn the patterns for user response classes beyond a point due to high variance. We need to optimize the algorithm for defining user response variable.

### Acknowledgement

This project has been as great learning experience as part of Udacity Data Scientist Nanodegree Program. Thanks to all mentors and peers who have helped in all stages.
