
#install packages______________________________
install.packages('dplyr')
install.packages('magrittr')
install.packages('ggplot2')
install.packages('Amelia')
install.packages('ROSE')
install.packages('ROCR')
install.packages('cvAUC')
install.packages('randomForest')
install.packages('effects')
install.packages("rgl")
install.packages('foreach')
install.packages('pROC')
install.packages('caret')
install.packages('xgboost')
install.packages('DiagrammeR')


library(dplyr)
library(magrittr)
library(ggplot2)
library(lattice)
library(Amelia)
library(caret)
library(ROSE)
library(ROCR)
library(cvAUC)
library(xgboost)
library(DiagrammeR)


setwd("C:/Users/mypc/Desktop/Data Analytics/Final Project/Data")

# loading the data___________________________________
data_train<-read.csv('training.csv',header = TRUE)

data_test<-read.csv('test.csv',header = TRUE)

# data has row id which is not important so we'll remove it
data_train<-data_train[,-1]
data_test<-data_test[,-1]


# missing values mapped in our data
jpeg(file="missing2.jpeg")
missmap(data_train, main = "Missing values vs observed")
dev.off()


#Data preprocessing- check for distribution and imputations

# 1. age
ggplot(data = data_train,aes(age))+geom_histogram(col='red',fill='red')+
labs(title='Histogram of Age')
qqnorm(data_train$age, main = "Age")
qqline(data_train$age)
sum(is.na(data_train$age))
summary(data_train$age)

# 2. RevolvingUtilizationOfUnsecuredLines- (ration should be between 0 and 1)
hist(data_train$RevolvingUtilizationOfUnsecuredLines,col='steelblue')
summary(data_train$RevolvingUtilizationOfUnsecuredLines)
boxplot(data_train$RevolvingUtilizationOfUnsecuredLines)

# total no. of missing values
sum(is.na(data_train$RevolvingUtilizationOfUnsecuredLines))

# potential outliers
sum(data_train$RevolvingUtilizationOfUnsecuredLines>1)

# plot to check for normal distribution 
qqnorm(data_train$RevolvingUtilizationOfUnsecuredLines[data_train$RevolvingUtilizationOfUnsecuredLines<=1])
qqline(data_train$RevolvingUtilizationOfUnsecuredLines)

# not a normal distribution so we'll impute abnormal values with median

data_train$RevolvingUtilizationOfUnsecuredLines[data_train$RevolvingUtilizationOfUnsecuredLines>1]=0.15

ggplot(data = data_train,aes(RevolvingUtilizationOfUnsecuredLines))+geom_histogram(col='red',fill='red')+
  labs(title='Histogram of RevolvingUtilizationOfUnsecuredLines')

boxplot(data_train$RevolvingUtilizationOfUnsecuredLines)

# 3. NumberOfTime30.59DaysPastDueNotWorse imputations 
summary(data_train$NumberOfTime30.59DaysPastDueNotWorse)

# total no. of missing values 
sum(is.na(data_train$NumberOfTime30.59DaysPastDueNotWorse))

# plot for distribution 
ggplot(data = data_train,aes(NumberOfTime30.59DaysPastDueNotWorse))+geom_histogram(col='black',fill='red')+
  labs(title='Histogram Num of time past 30-59days')

table(data_train$NumberOfTime30.59DaysPastDueNotWorse)

#few values are 96 and 98 which is absurd, we'll replace it by 0

data_train$NumberOfTime30.59DaysPastDueNotWorse[data_train$NumberOfTime30.59DaysPastDueNotWorse>=96]<-0

ggplot(data = data_train,aes(NumberOfTime30.59DaysPastDueNotWorse))+geom_histogram(col='black',fill='red')+
  labs(title='Histogram Num of time past 30-59days')

# 4. DebtRatio
summary(data_train$DebtRatio)

boxplot(data_train$DebtRatio)
sum(is.na(data_train$DebtRatio))

#debt ratio values greater than 100K is somewhat absurd so we are removing it

data_train<-data_train[-which(data_train$DebtRatio>100000),]
summary(data_train$DebtRatio)

boxplot(data_train$DebtRatio)

# 5. MonthlyIncome 
sum(is.na(data_train$MonthlyIncome))

summary(data_train$MonthlyIncome)

# monthly income above 300000 is removed as it can effect the model fitting
data_train<-data_train[-which(data_train$MonthlyIncome>300000),]
data_train
ggplot(data = data_train,aes(MonthlyIncome))+geom_histogram(col='black',fill='red')+
  labs(title='Histogram of MonthlyIncome')

#there are missing values in monthly income that we are we'll impute with median
ggplot(data = data_train,aes(MonthlyIncome))+geom_histogram(col='black',fill='red')+
  labs(title='Histogram of MonthlyIncome')

data_train$MonthlyIncome[is.na(data_train$MonthlyIncome)]<-median(data_train$MonthlyIncome,na.rm = TRUE)
boxplot(data_train$MonthlyIncome)


# 6. NumberOfOpenCreditLinesAndLoans 

sum(is.na(data_train$NumberOfOpenCreditLinesAndLoans))

summary(data_train$NumberOfOpenCreditLinesAndLoans)

#there are no missing values 
#check for normal distribution

ggplot(data = data_train,aes(NumberOfOpenCreditLinesAndLoans))+geom_histogram(col='red',fill='green')+
  labs(title='Histogram of num of open credit lines')

boxplot(data_train$NumberOfOpenCreditLinesAndLoans)
# distribution is normal and so it doesn't need any imputation

# 7. NumberOfTimes90DaysLate imputations 

sum(is.na(data_train$NumberOfTimes90DaysLate))
summary(data_train$NumberOfTimes90DaysLate)

#there are no missing values
ggplot(data = data_train,aes(NumberOfTimes90DaysLate))+geom_histogram(col='red',fill='green')+
  labs(title='Histogram of num of open credit lines')

boxplot(data_train$NumberOfTimes90DaysLate)

# data_train$SeriousDlqin2yrs[data_train$NumberOfTimes90DaysLate,1]

table(data_train$NumberOfTimes90DaysLate)
# default above 90 seems weird and might be inserted by error
# we'll impute it by 0 as median is also 0.


data_train$NumberOfTimes90DaysLate[data_train$NumberOfTimes90DaysLate>90]<-0

ggplot(data = data_train,aes(NumberOfTimes90DaysLate))+geom_histogram(col='red',fill='green')+
  labs(title='Histogram of num of open credit lines')

# 8. NumberRealEstateLoansOrLines imputations
summary(data_train$NumberRealEstateLoansOrLines)
sum(is.na(data_train$NumberRealEstateLoansOrLines))
table(data_train$NumberRealEstateLoansOrLines)

# there is value 54 which is a potential outlier that we'll remove
data_train<-data_train[-(which(data_train$NumberRealEstateLoansOrLines==54)),]

ggplot(data = data_train,aes(NumberRealEstateLoansOrLines))+geom_histogram(col='red',fill='green')+
  labs(title='Histogram of num of real estate loans')

# 9. NumberOfTime60.89DaysPastDueNotWorse imputations

sum(is.na(data_train$NumberOfTime60.89DaysPastDueNotWorse))

ggplot(data = data_train,aes(NumberOfTime60.89DaysPastDueNotWorse))+geom_histogram(col='red',fill='green')+
  labs(title='Histogram of num of times 60-89 days')
table(data_train$NumberOfTime60.89DaysPastDueNotWorse)

summary(data_train$NumberOfTime60.89DaysPastDueNotWorse)


# there are some value that seems to be absurd, we'll impute by 0
data_train$NumberOfTime60.89DaysPastDueNotWorse[data_train$NumberOfTime60.89DaysPastDueNotWorse>90]<-0

ggplot(data = data_train,aes(NumberOfTime60.89DaysPastDueNotWorse))+geom_histogram(col='red',fill='green')+
  labs(title='Histogram of num of times 60-89 days')

# 10. NumberOfDependents imputations 

sum(is.na(data_train$NumberOfDependents))

summary(data_train$NumberOfDependents)
# there are certain missing values that we'll impute with 0

data_train$NumberOfDependents[is.na(data_train$NumberOfDependents)]<-0

# Check if data is balanced?

prop.table(table(data_train$SeriousDlqin2yrs))
barplot(prop.table(table(data_train$SeriousDlqin2yrs)),col = 'steelblue')

# we can see serious data imbalance here that is 93% and 7% so we'll do downsampling for class'0' in variable SeriousDlqin2yrs
# we are making three set of samples so that all the observation have fair chances of random sampling

#first sample
sum(data_train$SeriousDlqin2yrs==1)
newdata_train<-data_train[data_train$SeriousDlqin2yrs==1,]
DownsampleData<-data_train[data_train$SeriousDlqin2yrs==0,]
downsample<-sample(1:139948,11000)
downsample

nData<-rbind(newdata_train,DownsampleData[downsample,])
nData<-nData[sample(nrow(nData)),]
rownames(nData)<-NULL

set.seed(36)
trainingIndex <- createDataPartition(nData$SeriousDlqin2yrs, p = .8, list = FALSE, times = 1)
ntraining<-nData[trainingIndex,]
ntesting<-nData[-trainingIndex,]

#second sample

set.seed(1234)

downsamplenew<-sample(1:139948,11000)
downsamplenew

nDatanew<-rbind(newdata_train,DownsampleData[downsamplenew,])
nDatanew<-nDatanew[sample(nrow(nData)),]
rownames(nDatanew)<-NULL

set.seed(40)
trainingIndex <- createDataPartition(nData$SeriousDlqin2yrs, p = .7, list = FALSE, times = 1)
ntrainingnew<-nDatanew[trainingIndex,]
ntestingnew<-nDatanew[-trainingIndex,]

# third sample
set.seed(2222)

downsamplenew1<-sample(1:139948,11000)
downsamplenew1

nDatanew1<-rbind(newdata_train,DownsampleData[downsamplenew1,])
nDatanew1<-nDatanew1[sample(nrow(nData)),]
rownames(nDatanew1)<-NULL

set.seed(50)
trainingIndex <- createDataPartition(nData$SeriousDlqin2yrs, p = .75, list = FALSE, times = 1)
ntrainingnew1<-nDatanew1[trainingIndex,]
ntestingnew1<-nDatanew1[-trainingIndex,]



# Data viz- melting of data frame__________________________________________________________________

library(reshape2)

feature.names<-names(nData)[-1]

vizData<- melt(nData,id.vars = 'SeriousDlqin2yrs'
              ,measure.vars = feature.names, variable.name = "Feature"
              ,value.name = "Value")

# conditional box plots for each feature on the response variable
p <- ggplot(data = vizData, aes(x=Feature, y=Value)) + 
  geom_boxplot(aes(fill=SeriousDlqin2yrs))
p <- p + facet_wrap( ~ Feature, scales="free")
p + ggtitle("Conditional Distributions of each variable")


#box plots- conditional distribtion of variables between classes 0 and 1___________________________________________
library("ggpubr")
ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "age", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "age", xlab = "Groups")

ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "RevolvingUtilizationOfUnsecuredLines", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "RevolvingUtilizationOfUnsecuredLines", xlab = "Groups")

ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "NumberOfTime30.59DaysPastDueNotWorse", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "NumberOfTime30.59DaysPastDueNotWorse", xlab = "Groups")

ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "DebtRatio", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "DebtRatio", xlab = "Groups")

ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "MonthlyIncome", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "MonthlyIncome", xlab = "Groups")


ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "NumberOfOpenCreditLinesAndLoans", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "NumberOfOpenCreditLinesAndLoans", xlab = "Groups")


ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "NumberOfTimes90DaysLate", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "NumberOfTimes90DaysLate", xlab = "Groups")


ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "NumberRealEstateLoansOrLines", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "NumberRealEstateLoansOrLines", xlab = "Groups")

ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "NumberOfTime60.89DaysPastDueNotWorse", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "NumberOfTime60.89DaysPastDueNotWorse", xlab = "Groups")


ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "NumberOfDependents", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "NumberOfDependents", xlab = "Groups")



#null Hypothesis______________________________________________________________________________________


install.packages('ggpubr')
install.packages('BSDA')
library(class)
library(ggpubr)
library(BSDA)
class(SeriousDlqin2yrs)
ntraining
attach(ntraining)
names(ntraining)


nrow(ntraining)
defaulted=ntraining[ntraining$SeriousDlqin2yrs==1,]
nrow(defaulted)
nondefaulted=ntraining[ntraining$SeriousDlqin2yrs==0,]
nrow(nondefaulted)


monthlyincome1=as.numeric(defaulted$MonthlyIncome)
monthlyincome2=as.numeric(nondefaulted$MonthlyIncome)
res1 <- z.test(monthlyincome1, monthlyincome2, alternative="less",mu=0,sigma.x = sd(monthlyincome2),sigma.y = sd(monthlyincome1),conf.level=0.95)
res1

monthlyincome1=as.numeric(defaulted$MonthlyIncome)
monthlyincome2=as.numeric(nondefaulted$MonthlyIncome)
res1 <- z.test(monthlyincome1, monthlyincome2, alternative="greater",mu=0,sigma.x = sd(monthlyincome2),sigma.y = sd(monthlyincome1),conf.level=0.95)
res1

#Boxplot

ggboxplot(ntraining, x = "SeriousDlqin2yrs", y = "MonthlyIncome", 
          color = "SeriousDlqin2yrs", palette = c("#00AFBB", "#E7B800"),
          ylab = "MonthlyIncome", xlab = "Groups")


#________________________________________________________________
# models
#________________________________________________________________


# logistic regression ___________________________________________

#Sample  1
library(effects)

# Main effect models 
logical.model<-glm(SeriousDlqin2yrs~.,data = ntraining,family = binomial)
summary(logical.model)
plot(allEffects(logical.model))

# ROC and AUROC
predict_logicalmodel<-predict(logical.model,ntesting[,-1],type='response')
predict_logicalmodel
predictLM <- prediction(predict_logicalmodel, ntesting$SeriousDlqin2yrs)
predictLM

install.packages('AUC')
library(ROCR)
library(AUC)
performLM <- performance(predictLM, measure = "tpr", x.measure = "fpr")
performLM
plot(performLM)
aucLM <- performance(predictLM, measure = "auc")
aucLM1 <- aucLM@y.values[[1]]
aucLM1

#Sample  2

logical.model<-glm(SeriousDlqin2yrs~.,data = ntrainingnew,family = binomial)
summary(logical.model)
plot(allEffects(logical.model))

# ROC and AUROC
predict_logicalmodel<-predict(logical.model,ntestingnew[,-1],type='response')
predict_logicalmodel
predictLM <- prediction(predict_logicalmodel, ntestingnew$SeriousDlqin2yrs)
predictLM
performLM <- performance(predictLM, measure = "tpr", x.measure = "fpr")
performLM
plot(performLM)
aucLM <- performance(predictLM, measure = "auc")
aucLM1 <- aucLM@y.values[[1]]
aucLM1


#Sample 3

logical.model<-glm(SeriousDlqin2yrs~.,data = ntrainingnew1,family = binomial)
summary(logical.model)
plot(allEffects(logical.model))

# ROC and AUROC
predict_logicalmodel<-predict(logical.model,ntestingnew1[,-1],type='response')
predict_logicalmodel
predictLM <- prediction(predict_logicalmodel, ntestingnew1$SeriousDlqin2yrs)
predictLM
performLM <- performance(predictLM, measure = "tpr", x.measure = "fpr")
performLM
plot(performLM)
aucLM <- performance(predictLM, measure = "auc")
aucLM1 <- aucLM@y.values[[1]]
aucLM1

# full interaction model
logical.interactionmodel<-glm(SeriousDlqin2yrs~(.)^2,data = ntraining,family = binomial)
summary(logical.interactionmodel)

anova(logical.interactionmodel,test = 'Chisq')

install.packages('car')
install.packages('stepwise')
library(car)
library(stepwise)

# step-wise selection 
logical.stepwise<-step(logical.interactionmodel, trace = 10)
summary(logical.stepwise)
# as we can see from the output lot of interaction are selected
anova(logical.stepwise, logical.interactionmodel, test='Cp')


library(cvAUC)
# ROC and AUROC FOR THE STEP-WISE MODEL
predict_logicalstepwise<-predict(logical.stepwise,ntesting[,-1],type='response')
summary(predict_logicalstepwise)
out<-cvAUC(predict_logicalstepwise,ntesting$SeriousDlqin2yrs)
plot(out$perf)
out$cvAUC
predictSLM <- prediction(predict_logicalmodel, ntesting$SeriousDlqin2yrs)
summary(predictSLM)
perfSLM <- performance(predictSLM, measure = "tpr", x.measure = "fpr")
summary(perfSLM)

#interaction between variables__________
head(ntraining)

interactionLM1<-glm(SeriousDlqin2yrs~RevolvingUtilizationOfUnsecuredLines + 
                      age + NumberOfTime30.59DaysPastDueNotWorse + DebtRatio + 
                      MonthlyIncome + NumberOfOpenCreditLinesAndLoans + NumberOfTimes90DaysLate + 
                      NumberRealEstateLoansOrLines + NumberOfTime60.89DaysPastDueNotWorse + 
                      NumberOfDependents+ RevolvingUtilizationOfUnsecuredLines:age, data=ntraining.gbm,
                    family=binomial)

summary(interactionLM1)

install.packages('effects')
library(effects)
plot(allEffects(focal.predictors = c("RevolvingUtilizationOfUnsecuredLines","NumberOfTime30.59DaysPastDueNotWorse"),interactionLM1,type='response'),multiline = FALSE)

interactionlogicalmodel2<- glm(SeriousDlqin2yrs~RevolvingUtilizationOfUnsecuredLines + age + NumberOfTime30.59DaysPastDueNotWorse + DebtRatio + 
    MonthlyIncome + NumberOfOpenCreditLinesAndLoans + NumberOfTimes90DaysLate + NumberRealEstateLoansOrLines + NumberOfTime60.89DaysPastDueNotWorse + 
  NumberOfDependents+MonthlyIncome:NumberRealEstateLoansOrLines, data=ntraining.gbm,family=binomial)
summary(interactionlogicalmodel2)

plot(Effect(focal.predictors = c("MonthlyIncome","NumberRealEstateLoansOrLines")
            ,interactionlogicalmodel2),type='response')

interactionlogicalmodel3<- glm(SeriousDlqin2yrs~RevolvingUtilizationOfUnsecuredLines + age + NumberOfTime30.59DaysPastDueNotWorse + DebtRatio + 
                                 MonthlyIncome + NumberOfOpenCreditLinesAndLoans + NumberOfTimes90DaysLate + NumberRealEstateLoansOrLines + NumberOfTime60.89DaysPastDueNotWorse + 
                                 NumberOfDependents+MonthlyIncome:NumberRealEstateLoansOrLines, data=ntraining.gbm,family=binomial)
summary(interactionlogicalmodel3)
plot(Effect(focal.predictors = c("age","MonthlyIncome")
            ,interactionlogicalmodel3),type='response')

interactionlogicalmodel4<- glm(SeriousDlqin2yrs~RevolvingUtilizationOfUnsecuredLines + age + NumberOfTime30.59DaysPastDueNotWorse + DebtRatio + 
                                 MonthlyIncome + NumberOfOpenCreditLinesAndLoans + NumberOfTimes90DaysLate + NumberRealEstateLoansOrLines + NumberOfTime60.89DaysPastDueNotWorse + 
                                 NumberOfDependents+MonthlyIncome:NumberRealEstateLoansOrLines, data=ntraining.gbm,family=binomial)
summary(interactionlogicalmodel4)

plot(Effect(focal.predictors = c("MonthlyIncome","NumberOfTime60.89DaysPastDueNotWorse")
            ,interactionlogicalmodel4),type='response')
# # #
plot(Effect(focal.predictors = c("NumberOfOpenCreditLinesAndLoans","NumberRealEstateLoansOrLines")
            ,interactionlogicalmodel3),type='response')
# # #
plot(Effect(focal.predictors = c("age","NumberOfTime60.89DaysPastDueNotWorse")
            ,interactionlogicalmodel3),type='response')
###
plot(Effect(focal.predictors = c("NumberOfDependents","MonthlyIncome")
            ,interactionlogicalmodel3),type='response')

library(ggplot2)
p <- ggplot(nData, aes(MonthlyIncome,NumberOfDependents ))
p + geom_point()



# random forest______________________________________________________
#sample 1
library(randomForest)

names(ntraining)<-c('SeriousDlqin2yrs','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10')
names(ntesting)<-c('SeriousDlqin2yrs','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10')
names(data_test)<-c('SeriousDlqin2yrs','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10')

a<-proc.time()

ntraining[,1]=as.factor(ntraining[,1])
ntesting[,1]=as.factor(ntesting[,1])

model <- randomForest(SeriousDlqin2yrs~., data=ntraining,do.trace=TRUE,importance=TRUE,ntree=500,mtry=3,forest=TRUE)
rf = randomForest(x=ntraining[,-1],y=ntraining$SeriousDlqin2yrs,ntree = 500,do.trace=TRUE,importance=TRUE)
pred.forest<-predict(rf,newdata = ntesting[,-1],na.action = na.pass)

cm1 = confusionMatrix(pred.forest,ntesting[,1],positive="1")
cm1
table(ntesting[,1],pred.forest)

proc.time()-a
varImpPlot(model)

pred.forest<-predict(model,newdata = ntesting[,-1],'prob')
output<-pred.forest[,2]
pr <- prediction(output, ntesting$SeriousDlqin2yrs)
prf2 <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf2)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#sample 2

names(ntrainingnew)<-c('SeriousDlqin2yrs','D1','D2','D3','D4','D5'
                    ,'D6','D7','D8','D9','D10')
names(ntestingnew)<-c('SeriousDlqin2yrs','D1','D2','D3','D4','D5'
                   ,'D6','D7','D8','D9','D10')
names(data_test)<-c('SeriousDlqin2yrs','D1','D2','D3','D4','D5'
                    ,'D6','D7','D8','D9','D10')

a<-proc.time()

ntrainingnew[,1]=as.factor(ntrainingnew[,1])
ntestingnew[,1]=as.factor(ntestingnew[,1])


model <- randomForest(SeriousDlqin2yrs~., data=ntrainingnew,do.trace=TRUE,importance=TRUE,ntree=500,mtry=3,forest=TRUE)
rf = randomForest(x=ntrainingnew[,-1],y=ntrainingnew$SeriousDlqin2yrs,ntree = 500,do.trace=TRUE,importance=TRUE)
pred.forest<-predict(rf,newdata = ntestingnew[,-1],na.action = na.pass)

cm1 = confusionMatrix(pred.forest,ntestingnew[,1],positive="1")
cm1

table(ntestingnew[,1],pred.forest)

proc.time()-a



varImpPlot(model)

pred.forest<-predict(model,newdata = ntestingnew[,-1],'prob')
output<-pred.forest[,2]
pr <- prediction(output, ntestingnew$SeriousDlqin2yrs)
prf2 <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf2)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#sample 3

names(ntrainingnew1)<-c('SeriousDlqin2yrs','D1','D2','D3','D4','D5'
                       ,'D6','D7','D8','D9','D10')
names(ntestingnew1)<-c('SeriousDlqin2yrs','D1','D2','D3','D4','D5'
                      ,'D6','D7','D8','D9','D10')
names(data_test)<-c('SeriousDlqin2yrs','D1','D2','D3','D4','D5'
                    ,'D6','D7','D8','D9','D10')

a<-proc.time()

ntrainingnew1[,1]=as.factor(ntrainingnew1[,1])
ntestingnew1[,1]=as.factor(ntestingnew1[,1])


model <- randomForest(SeriousDlqin2yrs~., data=ntrainingnew1,do.trace=TRUE,importance=TRUE,ntree=500,mtry=3,forest=TRUE)
rf = randomForest(x=ntrainingnew1[,-1],y=ntrainingnew1$SeriousDlqin2yrs,ntree = 500,do.trace=TRUE,importance=TRUE)
pred.forest<-predict(rf,newdata = ntestingnew1[,-1],na.action = na.pass)

cm1 = confusionMatrix(pred.forest,ntestingnew1[,1],positive="1")
cm1

table(ntestingnew1[,1],pred.forest)

proc.time()-a



varImpPlot(model)

pred.forest<-predict(model,newdata = ntestingnew1[,-1],'prob')
output<-pred.forest[,2]
pr <- prediction(output, ntestingnew1$SeriousDlqin2yrs)
prf2 <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf2)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc



##Decision Tree model__________________________________________________
#sample 1
require(rpart)
library(rpart)
install.packages('klaR')
library(klaR)
install.packages('party')
library(party)

ntraining[,1]=as.factor(ntraining[,1])
ntesting[,1]=as.factor(ntesting[,1])

output.tree <- ctree(SeriousDlqin2yrs ~ D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10, data = ntraining)
plot(output.tree)
model.dT <- rpart(formula = SeriousDlqin2yrs ~.,data = ntraining,method = 'class')

##Prediction Tree model

pred.decision<-predict(model.dT,newdata = ntesting[,-1],'prob')
output<-pred.decision[,2]
pr <- prediction(output, ntesting$SeriousDlqin2yrs)
prf2 <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf2)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

model.dT

pred.decisiontree<-predict(output.tree,newdata = ntesting[,-1],na.action = na.pass)
table(ntesting[,1],pred.decisiontree)
cm = confusionMatrix(pred.decisiontree,ntesting[,1],positive="1")
cm


#sample 2

ntrainingnew[,1]=as.factor(ntrainingnew[,1])
ntestingnew[,1]=as.factor(ntestingnew[,1])

output.tree <- ctree(SeriousDlqin2yrs ~ D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10, data = ntrainingnew)
plot(output.tree)
model.dT <- rpart(formula = SeriousDlqin2yrs ~.,data = ntrainingnew,method = 'class')

##Prediction Tree model

pred.decision<-predict(model.dT,newdata = ntestingnew[,-1],'prob')
output<-pred.decision[,2]
pr <- prediction(output, ntestingnew$SeriousDlqin2yrs)
prf2 <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf2)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

model.dT

pred.decisiontree<-predict(output.tree,newdata = ntestingnew[,-1],na.action = na.pass)
table(ntestingnew[,1],pred.decisiontree)


cm = confusionMatrix(pred.decisiontree,ntestingnew[,1],positive="1")
cm

#sample 3

ntrainingnew1[,1]=as.factor(ntrainingnew1[,1])
ntestingnew1[,1]=as.factor(ntestingnew1[,1])

output.tree <- ctree(SeriousDlqin2yrs ~ D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10, data = ntrainingnew1)
plot(output.tree)
model.dT <- rpart(formula = SeriousDlqin2yrs ~.,data = ntrainingnew1,method = 'class')

##Prediction Tree model

pred.decision<-predict(model.dT,newdata = ntestingnew1[,-1],'prob')
output<-pred.decision[,2]
pr <- prediction(output, ntestingnew1$SeriousDlqin2yrs)
prf2 <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf2)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

model.dT

pred.decisiontree<-predict(output.tree,newdata = ntestingnew1[,-1],na.action = na.pass)
table(ntestingnew1[,1],pred.decisiontree)


cm = confusionMatrix(pred.decisiontree,ntestingnew1[,1],positive="1")
cm


#KNN model________________________________
# Sample 1

library(class)

predictKNN=knn(train=ntraining[,-1], test=ntesting[,-1], cl=ntraining[,1])
predictKNN

KNNpr <- prediction(as.numeric(predictKNN), as.numeric(ntesting$SeriousDlqin2yrs))
KNNpr
KNNprf <- performance(KNNpr, measure = "tpr", x.measure = "fpr")
KNNprf
plot(KNNprf)

aucKNN <- performance(KNNpr, measure = "auc")
aucKNN <- aucKNN@y.values[[1]]
aucKNN

ConfusionKNN=confusionMatrix(predictKNN, ntesting[,1])
ConfusionKNN


#KNN after scaling

library(lattice)
library(ggplot2)

set.seed(300)

IndexTrain<- createDataPartition(y = ntraining$SeriousDlqin2yrs,p = 0.75,list = FALSE)
ktrain<- ntraining[IndexTrain,]
ktest<- ntraining[-IndexTrain,]
prop.table(table(ktrain$SeriousDlqin2yrs)) * 100
prop.table(table(ktest$SeriousDlqin2yrs)) * 100
prop.table(table(ntraining$SeriousDlqin2yrs)) * 100


trainX <- ktrain[,names(ktrain) != "SeriousDlqin2yrs"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues


set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3)#, classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(SeriousDlqin2yrs ~ ., data = ktrain, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 10)
knnFit

plot(knnFit)


knnPredict <- predict(knnFit,newdata = ktest )
knnPredict
confusionMatrix(knnPredict,ktest$SeriousDlqin2yrs )
mean(knnPredict == ktest$SeriousDlqin2yrs)


library(pROC)

ctrl <- trainControl(method="repeatedcv",repeats = 3, classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(SeriousDlqin2yrs ~ ., data = ktrain, method = "knn", trControl = ctrl, preProcess = c("center","scale"),tuneLength = 10)
KnnFit
knnPredict <- predict(knnFit,newdata = ktest , type="prob")
knnpredict
knnROC <- roc(ktest$SeriousDlqin2yrs,knnPredict[,"0"],levels = rev(ktest$SeriousDlqin2yrs))
knnROC

plot(knnROC, type="S", print.thres= 0.5)

KNNpr1 <- prediction(as.numeric(knnPredict), as.numeric(ktest$SeriousDlqin2yrs))
KNNpr1
KNNprf1 <- performance(KNNpr, measure = "tpr", x.measure = "fpr")
KNNprf1
plot(KNNprf1)

aucKNN1 <- performance(KNNpr1, measure = "auc")
aucKNN1 <- aucKNN1@y.values[[1]]
aucKNN1

#sample 2
set.seed(300)

IndexTrain<- createDataPartition(y = ntrainingnew$SeriousDlqin2yrs,p = 0.75,list = FALSE)
ktrain<- ntrainingnew[IndexTrain,]
ktest<- ntrainingnew[-IndexTrain,]
prop.table(table(ktrain$SeriousDlqin2yrs)) * 100
prop.table(table(ktest$SeriousDlqin2yrs)) * 100
prop.table(table(ntrainingnew$SeriousDlqin2yrs)) * 100


trainX <- ktrain[,names(ktrain) != "SeriousDlqin2yrs"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues


set.seed(400)
ctrl <- trainControl(method="repeatedcv",repeats = 3)#, classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(SeriousDlqin2yrs ~ ., data = ktrain, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 10)
knnFit

plot(knnFit)


knnPredict <- predict(knnFit,newdata = ktest )
knnPredict
confusionMatrix(knnPredict,ktest$SeriousDlqin2yrs )
mean(knnPredict == ktest$SeriousDlqin2yrs)


library(pROC)

ctrl <- trainControl(method="repeatedcv",repeats = 3, classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(SeriousDlqin2yrs ~ ., data = ktrain, method = "knn", trControl = ctrl, preProcess = c("center","scale"),tuneLength = 10)
KnnFit
knnPredict <- predict(knnFit,newdata = ktest , type="prob")
knnpredict
knnROC <- roc(ktest$SeriousDlqin2yrs,knnPredict[,"0"],levels = rev(ktest$SeriousDlqin2yrs))
knnROC

plot(knnROC, type="S", print.thres= 0.5)

KNNpr1 <- prediction(as.numeric(knnPredict), as.numeric(ktest$SeriousDlqin2yrs))
KNNpr1
KNNprf1 <- performance(KNNpr, measure = "tpr", x.measure = "fpr")
KNNprf1
plot(KNNprf1)

aucKNN1 <- performance(KNNpr1, measure = "auc")
aucKNN1 <- aucKNN1@y.values[[1]]
aucKNN1

##naive bayes_______________________________________________

install.packages('naivebayes', dependencies=TRUE, repos='http://cran.rstudio.com/')
install.packages('dplyr')
install.packages('ggplot2')
install.packages('psych')
install.packages("pROC")
install.packages('ROCR')

library(ROCR)
library(pROC)
library(naivebayes)
library(dplyr)
library(ggplot2)
library(psych)
library(caret)




# sample 1

NaiveBayesModel<-naive_bayes(SeriousDlqin2yrs ~ ., data=ntraining)
NaiveBayesModel
plot(NaiveBayesModel)
#Predict
pred<-predict(NaiveBayesModel,ntraining,type='prob')
head(cbind(pred,ntraining))
#Confusion Matrix-train data
p1<- predict(NaiveBayesModel,ntraining)
(tab1<-table(p1,ntraining$SeriousDlqin2yrs))
1 - sum(diag(tab1))/sum(tab1)  #Missclass
confusionMatrix(tab1)
#Confusion Matrix-test data
p2<- predict(NaiveBayesModel,ntesting)
(tab2<-table(p2,ntesting$SeriousDlqin2yrs))
1 - sum(diag(tab2))/sum(tab2) 
confusionMatrix(tab2)

# ROC and AUROC
pred.NaiveBayes<-predict(NaiveBayesModel,ntesting[,-1],type='prob')
pred.NaiveBayes
oput<-pred.NaiveBayes[,2]
pr01 <- prediction(as.numeric(oput), as.numeric(ntesting$SeriousDlqin2yrs))
pr01
prf01 <- performance(pr01, measure = "tpr", x.measure = "fpr")
plot(prf01)
plot(prf01, main = "ROC curve for Naive Bayes Classifier",col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
perf.auc <- performance(pr01, measure = "auc")
unlist(perf.auc@y.values)




trainX <- ntraining[,names(ntraining) != "SeriousDlqin2yrs"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues
#set.seed(400)
#ctrl<-trainControl(method="repeatedcv")
NaiveBayesModel<-naive_bayes(SeriousDlqin2yrs ~ ., data=ntraining,preProcess = c("center","scale"))
NaiveBayesModel
plot(NaiveBayesModel)
#Predict
pred<-predict(NaiveBayesModel,ntraining,type='prob')
head(cbind(pred,ntraining))
#Confusion Matrix-train data
p1<- predict(NaiveBayesModel,ntraining)
(tab1<-table(p1,ntraining$SeriousDlqin2yrs))
1 - sum(diag(tab1))/sum(tab1)  
confusionMatrix(tab1)
#Confusion Matrix-test data
p2<- predict(NaiveBayesModel,ntesting)
(tab2<-table(p2,ntesting$SeriousDlqin2yrs))
1 - sum(diag(tab2))/sum(tab2) 
confusionMatrix(tab2)
# ROC and AUROC
pred.NaiveBayes<-predict(NaiveBayesModel,ntesting[,-1],type='prob')
pred.NaiveBayes
oput1<-pred.NaiveBayes[,2]
pr01 <- prediction(as.numeric(oput1), as.numeric(ntesting$SeriousDlqin2yrs))
pr01
prf01 <- performance(pr01, measure = "tpr", x.measure = "fpr")
plot(prf01)
plot(prf01, main = "ROC curve for Naive Bayes Classifier",col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
perf.auc <- performance(pr01, measure = "auc")
unlist(perf.auc@y.values)

#Sample 2
trainX <- ntrainingnew[,names(ntrainingnew) != "SeriousDlqin2yrs"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues

NaiveBayesModel<-naive_bayes(SeriousDlqin2yrs ~ ., data=ntrainingnew,preProcess = c("center","scale"))
NaiveBayesModel
plot(NaiveBayesModel)
#Predict
pred<-predict(NaiveBayesModel,ntrainingnew,type='prob')
head(cbind(pred,ntrainingnew))
#Confusion Matrix-train data
p1<- predict(NaiveBayesModel,ntrainingnew)
(tab1<-table(p1,ntrainingnew$SeriousDlqin2yrs))
1 - sum(diag(tab1))/sum(tab1)  
confusionMatrix(tab1)
#Confusion Matrix-test data
p2<- predict(NaiveBayesModel,ntestingnew)
(tab2<-table(p2,ntestingnew$SeriousDlqin2yrs))
1 - sum(diag(tab2))/sum(tab2) 
confusionMatrix(tab2)
# ROC and AUROC
pred.NaiveBayes<-predict(NaiveBayesModel,ntestingnew[,-1],type='prob')
pred.NaiveBayes
oput1<-pred.NaiveBayes[,2]
pr01 <- prediction(as.numeric(oput1), as.numeric(ntestingnew$SeriousDlqin2yrs))
pr01
prf01 <- performance(pr01, measure = "tpr", x.measure = "fpr")
plot(prf01)
plot(prf01, main = "ROC curve for Naive Bayes Classifier",col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
perf.auc <- performance(pr01, measure = "auc")
unlist(perf.auc@y.values)

#Sample 3
trainX <- ntrainingnew1[,names(ntrainingnew1) != "SeriousDlqin2yrs"]
preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
preProcValues

NaiveBayesModel<-naive_bayes(SeriousDlqin2yrs ~ ., data=ntrainingnew1,preProcess = c("center","scale"))
NaiveBayesModel
plot(NaiveBayesModel)
#Predict
pred<-predict(NaiveBayesModel,ntrainingnew1,type='prob')
head(cbind(pred,ntrainingnew1))
#Confusion Matrix-train data
p1<- predict(NaiveBayesModel,ntrainingnew1)
(tab1<-table(p1,ntrainingnew1$SeriousDlqin2yrs))
1 - sum(diag(tab1))/sum(tab1)  
confusionMatrix(tab1)
#Confusion Matrix-test data
p2<- predict(NaiveBayesModel,ntestingnew1)
(tab2<-table(p2,ntestingnew1$SeriousDlqin2yrs))
1 - sum(diag(tab2))/sum(tab2) 
confusionMatrix(tab2)
# ROC and AUROC
pred.NaiveBayes<-predict(NaiveBayesModel,ntestingnew1[,-1],type='prob')
pred.NaiveBayes
oput1<-pred.NaiveBayes[,2]
pr01 <- prediction(as.numeric(oput1), as.numeric(ntestingnew1$SeriousDlqin2yrs))
pr01
prf01 <- performance(pr01, measure = "tpr", x.measure = "fpr")
plot(prf01)
plot(prf01, main = "ROC curve for Naive Bayes Classifier",col = "blue", lwd = 3)
abline(a = 0, b = 1, lwd = 2, lty = 2)
perf.auc <- performance(pr01, measure = "auc")
unlist(perf.auc@y.values)

