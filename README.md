# ML_Project "Facebook"

This is a machine learning project using classifiers to predict whether person will click to watch the add on Facebook or not, using data from CSV file

A several visual aids like graphs and confusion matrix have been created, to show and analyze the results.

## Table Of Content
- [Project](#Project)
    - [About dataset](#About_Dataset)
    - [Scope](#Scope)
    - [Data preparation](#Data_Preparation)
    - [Model development](#Model_Development)
- [Libraries and Algorithms used](#Liabraries_and_Algorithms)
    - [Libraries](#Libraries_and_Modules)
    - [Algorithms](#Algorithms)

## Project

Current project was developed within the framework of study AI. The main point was to perform data preprocessing, ML model training and model hyper parameters tuning via GridSearch.

### About_Dataset

In this Project, a dataset that has users from different countries of both genders and also determining factors like, salary range, number of hours spent on site etc. is used, which is a CSV File.

### Scope

Project scope:

*   perform data preparation
*   visualize data
*   model development
    
### Data_Preparation

    - convert name variable to gender
    - perform outliers treatment (1. by replacing outliers with median value/ 2. by trimming outliers)
    - perform encoding
    - perform discretization 
    - perform normalization
    - perform not relevant variables removal
    
### Model_Development
 
    - by choosing optimal Algorithm
    - by hyper parameters tuning via grid search

## Libraries_and_Algorithms
### Libraries_and_Modules

libraries and modules, their implementation in current project:

*   [pandas](https://pandas.pydata.org/) - for data manipulation and analysis
*   [numpy](https://numpy.org/) - for scientific computing with Python
*   [matplotlib](https://matplotlib.org/) - for creating static, animated, and interactive visualizations in Python
*   [seaborn](https://seaborn.pydata.org/) - for drawing attractive and informative statistical graphics
*   [sklearn](https://scikit-learn.org/stable/) - to analyze results
*   [scipy](https://scipy.org/) - to visualize normal distribution with given parameters
*   [requests](https://pypi.org/project/requests/) - to send HTTP/1.1 requestto RestApi
*   [json](https://docs.python.org/3/library/json.html) - to encode and decode json
*   [feature_engine](https://feature-engine.readthedocs.io/en/latest/) - to descritize data
*   [statsmodels](https://www.statsmodels.org/stable/index.html) - to visualize results
*   [pipeline](https://scikit-learn.org/stable/) - to analyze results
*   [grid search](https://scikit-learn.org/stable/) - to analyze results

### Algorithms

*   [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - the probabilities describing the possible outcomes of a single trial are modeled using a logistic function
*   [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - C-Support Vector Classification
*   [GaussianProcessClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html) - based on Laplace approximation
*   [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) - implements the Gaussian Naive Bayes algorithm for classification
*   [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) - implements the k-nearest neighbors vote
*   [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) - by learning simple decision rules inferred from the data features, predicts the value of a target variable
*   [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
*   [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) - meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases
*   [AdaBoostClassifier](https://pandas.pydata.org/) - for data manipulation and analysis

