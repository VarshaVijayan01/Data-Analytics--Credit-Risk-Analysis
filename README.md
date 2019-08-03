# Data-Analytics - Credit Risk Analysis

Credit can be considered as the one of the key elements on which a market and society functions.The banks have the right to build and break the decisions based on investments, and they also make crucial decisions  based on the terms on which the bank provides people with credit. The decision based on whether a loan should be granted or not, is determined by the banks using the Credit Scoring Algorithm and this method helps in making a guess at the probability of default. This dataset is used towards improving the state of credit scoring,  by making a prediction that in the next two years there is a probability that certain bankers will face a financial distress.

This Dataset is from Kaggle- “ Give Me Some Credit”
Number of observations- 150,000 records.
Number of Variables-11 

## While exploring the data 33655 missing values were found. 

![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Rplot.png)
![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Missing%202.png)

## Each variable was carefully analyzed with the help of histogram to understand the distribution of each variable.

![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Overall%20%2001%20data%20exploration%20and%20cleaning.png)

![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Overall%20%2002%20data%20exploration%20and%20cleaning.png)

Variable RevolvingUtilizationOfUnsecuredLines had no missing values but it had 3321 outliersFrom the analysis it can be seen that the distribution is not a normal distribution so the abnormal values will be imputed with median. After imputing the distribution was better
Similarly in variable NumberOfTime30.59DaysPastDueNotWorse, it  had no missing values but it had 269 outliers and it was replaced with 0
Monthly income had 29723 missing values when looked into. This was solved my imputing median in these missing values.
Debt ratio values greater than 100K is somewhat absurd so we are removing it
No missing values for NumberOfTimes90DaysLate but there are some unusal patterns above 90 and therefore decided to remove it.
Again no missing value in NumberRealEstateLoansOrLines but outlier is seen and removed.
NumberOfTime60.89DaysPastDueNotWorse imputed absurd values with 0.
NumberOfDependents missing values were imputed with 0.

  On testing the balance proportion of the resultant value we can see an imbalance here that is 93% and 7% so we'll do downsampling for class'0' in variable SeriousDlqin2yrs to get a balance between the values
  
  ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Balance%20Testing.png)
  
Three set of samples were used so that all the observation have fair chance of random sampling
    
## Two sample One-tailed Hypothesis Testing
Null Hypothesis : The average age of the defaulters is less than the average of the non defaulters. 
To check hypothesis, t test has been performed where p value is less than 0.05. Thus, at 95% confidence interval, there is no enough evidence to reject null hypothesis
  ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Hypothesis%201.png)

Null Hypothesis : The average monthly income of the defaulters is less than the average of the non defaulters. 
To check hypothesis, t test has been done where p value is less than 0.05. Thus, at 95% confidence interval, there is no enough evidence to reject null hypothesis

  ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Hypothesis%202.png)
  
  
## MODELING
## Naive Bayes
    ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Naive%20bayes%201.png)
  
      ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Naive%20bayes%202.png.jpg)
      
The naïve bayes gives an accuracy of 85.94%. It is used in various field as it is considered easy to implement and provides accurate results.
  
## Decision Tree
    ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/DT.png)
  
      ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/DT%201.pnghttps://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Naive%20bayes%202.png.jpg)
 
 The decision tree gives an accuracy of 81.59%.
This model is a decision support tool that uses atree-like graph or model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements..

## Random Forest
          ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/RF.png)

          ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/RF2.png)

The random forest gives an accuracy of 85.94%.This model is used as it is one of the most commonly used model for classification to help the bank in making its financial decisions.
          
## KNN
AUC value of KNN model before scaling
          ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/knn1.png)
AUC value of KNN model before scaling
          ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/knn2.png)

          ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/knn3.png)
          
 Random Forest gives the highest accuracy of 85.9%
          
Overall Evaluation of all models
          ![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Evaluation.png)


