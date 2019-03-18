# Group 4
# Machine Learning Coursework 3

# SETUP -------------------------------------------------------------------

library(caret)
library(e1071)
library(ROCR)

# DATA --------------------------------------------------------------------

library(ISLR)

data("OJ")

# investigate the data
head(OJ)
str(OJ)

# StoreID, Store7 and STORE essentially have the same information (7 in StoreID is 0 in STORE )
OJ[,c("StoreID", "Store7", "STORE")]

# remove StoreID and Store 7
OJ = OJ[, !(colnames(OJ) %in% c("StoreID", "Store7"))]

# categorical variables to dummy variables
library(caret)
dmy = dummyVars(~ ., data = OJ)
dmy = data.frame(predict(dmy, newdata = OJ))

# split the data (x) and the labels (y)
data = dmy[,-(1:2)]
labels = OJ[,1]

# split the dataset into training and test sets
set.seed(123)
train.index = createDataPartition(labels, p = 0.7, list=F)

train.x = data[train.index,]
train.y = labels[train.index]

test.x = data[-train.index,]
test.y = labels[-train.index]


# 2.1 ---------------------------------------------------------------------

set.seed(000)

# tune the parameters for linear kernel
tune.out1 = tune(svm, train.x, train.y, 
                kernel= "linear",
                ranges = list(cost = 10^(-2:1)),
                tunecontrol = tune.control(sampling = "cross"), cross=10)

# check the best model
tune.out1$best.model 

# predict with the tuned value for cost
pred1=predict(tune.out1$best.model, test.x, decision.values = T) 
pred1.dv = attributes(pred1)$decision.values

# confusion matrix
table(pred1, test.y)

# test error rate
mean(pred1!=test.y)


# 2.2 ---------------------------------------------------------------------

set.seed(000)

# tune the parameters for radial kernel
tune.out2 = tune(svm, train.x, train.y, 
                kernel= "radial",
                ranges = list(cost = 10^(-2:1)),
                tunecontrol = tune.control(sampling = "cross"), cross=10)

# check the best model
tune.out2$best.model

# predict with the tuned value for cost
pred2=predict(tune.out2$best.model, test.x, decision.values = T) 
pred2.dv = attributes(pred2)$decision.values

# confusion matrix
table(pred2, test.y) 

# test error rate
mean(pred2!=test.y)


# 2.3 ---------------------------------------------------------------------

set.seed(000)

# tune the parameters for polynomial kernel (with degree=2)
tune.out3 = tune(svm, train.x, train.y, 
                 kernel= "polynomial",
                 degree=2,
                 ranges = list(cost = 10^(-2:1)),
                 tunecontrol = tune.control(sampling = "cross"), cross=10)

# check the best model
tune.out3$best.model

# predict with the tuned value for cost
pred3=predict(tune.out3$best.model, test.x, decision.values = T) 
pred3.dv = attributes(pred3)$decision.values

# confusion matrix
table(pred3, test.y)

# test error rate
mean(pred3!=test.y)


# 2.4 ---------------------------------------------------------------------

# Roc plot

source("rocplot.R")

rocplot(pred1.dv,test.y,order=c("MM","CH"),col="green",lwd=2,cex.lab=1,cex.axis=1,main="ROC Curves")
rocplot(pred2.dv,test.y,order=c("MM","CH"),col="red", add=TRUE, lwd=2,cex.lab=1.5,cex.axis=1.5)
rocplot(pred3.dv,test.y,order=c("MM","CH"),col="blue", add=TRUE, lwd=2,cex.lab=1.5,cex.axis=1.5)

legend("bottomright", cex=0.9,
       c("linear (C=0.01, γ=0.067)", "radial (C=1, γ=0.067)", "polynomial (C=10, γ=0.067, d=2)"),
       col=c("green","red","blue"),lty=1,lwd=2)
#dsjvfhjedks
