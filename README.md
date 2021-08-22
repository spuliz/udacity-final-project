# Udacity Machine Learning Capstone Project
This GitHub repository hosts the Capstone project I developed and completed as part of the Udacity Machine Learning Engineer Nanodegree.

## Domain Background
Arvato Financial Services is a company operating in the mail-order sales business in Germany and is a subsidiary of Bertelsmann.
The company wants to grow their customer base by better targeting clusters of the general population with their marketing campaigns.

## Table of contents
- [Notebook](https://github.com/spuliz/udacity-final-project/blob/main/Arvato_Project_Supervised_Learning.ipynb): main project jupyter notebook
- [Metadata](https://github.com/spuliz/udacity-final-project/tree/main/metadata): folder containing the metadata of the datasets 
- [Report](https://github.com/spuliz/udacity-final-project/tree/main/report): Folder contaning the initial projet proposal and the final report 
- [Utils](https://github.com/spuliz/udacity-final-project/tree/main/utils.py): Helper functions used to clean the datasets and retrieved from: https://github.com/AilingLiu/Machine-Learning-Engineer-Nanodegree-Program-Udacity

## Datasets
Demographics data of the general population and of prior customers of the business will be used in order to identify those individuals who are more likely to respond to the marketing campaign and to become customers of the mail-order company.
Datasets are available to use within the related Udacity Workspace and cannot be publicly shared in this repository.

The data has been provided by Bertelsmann Arvato Analytics, and consists of demographics data of the general population of Germany, for prior customers of the company and of individuals targeted on a marketing campaign. 

In particular, there are four datasets: 

- Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
- Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
- Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
- Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood.

The "CUSTOMERS" file contains three extra columns ('CUSTOMER_GROUP', 'ONLINE_PURCHASE', and 'PRODUCT_GROUP'), which provide broad information about the customers depicted in the file. 

The original "MAILOUT" file included one additional column, "RESPONSE", which indicated whether or not each recipient became a customer of the company. For the "TRAIN" subset, this column has been retained, but in the "TEST" subset it has been removed.


## The ML system
The overall system can be diveded in two parts:
- The first part where unsupervised learning methods will be used to analyze attributes of established customers and the general population in order to create clusters and compare them.
- The second part where the previous analysis will be used against a target dataset with attributes from customers targeted of a mail order campaign. A supervised learning approach will be used in order to build a machine learning model that predicts whether or not an individual will respond to the marketing campaign.

## Libraries used
The Jupyter Notebook is written in Python.

The main packages used are:

- numpy: scientific computing tools

- pandas: data structures and data analysis tools

- matplotlib: data visualisation tools

- seaborn: data visualisation tools

- scikit-learn (sklearn): Machine Learning library in Python


## Setup
install the packages listed in the file requirements.txt

# 4th best score in a Kaggle competition with 500+ teams!
![image](https://user-images.githubusercontent.com/15948985/130328547-e3ae022d-2915-4fdf-b4cf-92fd42b794a6.png)


## References and supporting materials
[1] pandas.get_dummies. In Pandas Documentation. Retrieved from:
https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

[2] Impute Missing Values with SciKit’s Imputer – Python. In Medium. Retrieved from:
https://medium.com/technofunnel/handling-missing-data-in-python-using-scikit-imputer7607c8957740

[3] StandardScaler. In Scikit-Learn Documentation. Retrieved from:
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

[4] Understanding Principal Component Analysis. In Medium. Retrieved from:
https://medium.com/@aptrishu/understanding-principle-component-analysis-e32be0253ef0

[5] K-Means Clustering in Python: A Practical Guide. In Real Python. Retrieved from:
https://realpython.com/k-means-clustering-python/

[6] Classification: ROC Curve and AUC. In Google Developers. Retrieved from:
https://www.datanovia.com/en/wp-content/uploads/dn-tutorials/002-partitionalclustering/images/partitioning-clustering.png

[7] RandomForestClassifier. In Scikit-Learn Documentation. Retrieved from:
https://scikitlearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

[8] Understanding Random Forests Classifiers in Python. In Datacamp Community. Retrieved
from: https://www.datacamp.com/community/tutorials/random-forests-classifier-python

[9] GradientBoostingClassifier. In Scikit-Learn Documentation. Retrieved from:
https://scikitlearn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

[10] Gradient Boosting with Scikit-Learn, XGBoost, LightGBM, and CatBoot. In Machine
Learning Mastery. Retrieved from: https://machinelearningmastery.com/gradient-boostingwith-scikit-learn-xgboost-lightgbm-and-catboost/

[11] Github repository: Machine-Learning-Engineer-Nanodegree-Program-Udacity. Retrived from: https://github.com/AilingLiu/Machine-Learning-Engineer-Nanodegree-Program-Udacity/

[12] Github repository: Arvato-MLProject. Retrieved from: https://github.com/dilayercelik/Arvato-MLProject

[13] SVC. In Scikit-Learn Documentation. Retrieved from:
https://scikitlearn.org/stable/modules/generated/sklearn.svm.SVC.html

[14] Elbow Method for Optimal Value k in K-Means. In GeeksforGeeks. Retrieved from:
https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/


## Author
Saverio Pulizzi

email: saverio.pulizzi91@gmail.com









