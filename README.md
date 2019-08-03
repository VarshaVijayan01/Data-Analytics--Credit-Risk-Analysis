Credit can be considered as the one of the key elements on which a market and society functions.The banks have the right to build and break the decisions based on investments, and they also make crucial decisions  based on the terms on which the bank provides people with credit. The decision based on whether a loan should be granted or not, is determined by the banks using the Credit Scoring Algorithm and this method helps in making a guess at the probability of default. We will be working towards improving the state of credit scoring,  by making a prediction that in the next two years there is a probability that certain bankers will face a financial distress.

This Dataset is from Kaggle- “ Give Me Some Credit”
Number of observations- 150,000 records.
Number of Variables-11 

While exploring the data 33655 missing values were found. 

![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Rplot.png)


![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Missing%202.png)

Each variable was carefully analyzed with the help of histogram to understand the distribution of each variable.


Variable RevolvingUtilizationOfUnsecuredLines had no missing values but it had 3321 outliers

![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Unsecured01.jpg)

![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/UnsecuredLines.png)

From the analysis it can be seen that the distribution is not a normal distribution so the abnormal values will be imputed with median. After imputing the distribution was better

![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/UnsecuredLines.png)


Similarly in variable NumberOfTime30.59DaysPastDueNotWorse, it  had no missing values but it had 269 outliers and it was replaced with 0

Monthly income had 29723 missing values when looked into.
![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Before%20monthly.jpg)
![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/Monthly%20income%20na.PNG)

This was solved my imputing median in these missing values.
![alt text](https://github.com/VarshaVijayan01/Data-Analytics--Credit-Risk-Analysis/blob/master/Images/aftermonthly.jpg)

debt ratio values greater than 100K is somewhat absurd so we are removing it
No missing values for NumberOfTimes90DaysLate but there are some unusal patterns above 90 and therefore decided to remove it.
Again no missing value in NumberRealEstateLoansOrLines but outlier is seen and removed.
NumberOfTime60.89DaysPastDueNotWorse imputed absurd values with 0.
NumberOfDependents missing values were imputed with 0.
