#Final Exam - Group 4
#George Garcia, Marco Duran Perez, Hilary Balli

####################
# Libraries and data

library(readr)
library(caret)
#library(AppliedPredictiveModeling)
library(corrplot)
library(skimr)
library(dplyr)
library(beepr)
library(plyr)
library(penalizedLDA)
final <- read.csv("C:/Users/george/Downloads/QUMT 6350 Downloads/examdata2.csv")



# 1. Identify case
head(final)
str(final)
summary(final)
# The outcome variable suggests a 2-class case of classification.



dataFinal <- final

x <- subset(dataFinal, select = -Risk1Yr)
y <- subset(dataFinal, select = Risk1Yr)

# Review imbalances in outcome variable
barplot(table(y), 
        names.arg = c("Died", "Survived"),
        col = c("blue", "green"),
        main = "Risk1Yr")
# The outcome variable has 400 observations of "Survived" and 70 observations of "Died", therefore
# this data set is imbalanced, and we will use stratified splitting.



# 2. set training and testing data
# Set seed
seed <- 250
set.seed(seed)

Ftrain <- createDataPartition(y$Risk1Yr,
                              p = .75,
                              list = FALSE)
xTest <- x[-Ftrain,]
yTest <- y[-Ftrain,]

xTrain <- x[Ftrain,]
yTrain <- y[Ftrain,]



# 3. explore data for training set
skimmed1 <- skim_to_wide(xTrain)
skimmed2 <- skim_to_wide(yTrain)
View(skimmed1)
View(skimmed2)

#set x variables to numeric
xTrainNum <- as.data.frame(sapply(xTrain[,1:16], as.numeric))

#boxplot of predictors
boxplot(xTrainNum)

# Histograms of predictors and outcome variables
par(mfrow = c(3,3))
hist(xTrainNum$Attribute)
hist(xTrainNum$PRE4)
hist(xTrainNum$PRE5)
hist(xTrainNum$PRE6)
hist(xTrainNum$PRE7)
hist(xTrainNum$PRE8)
hist(xTrainNum$PRE9)
hist(xTrainNum$PRE10)
hist(xTrainNum$PRE11)
hist(xTrainNum$PRE14)
hist(xTrainNum$PRE17)
hist(xTrainNum$PRE19)
hist(xTrainNum$PRE25)
hist(xTrainNum$PRE30)
hist(xTrainNum$PRE32)
hist(xTrainNum$AGE)
hist(as.numeric(yTrain))
# The training data has 17 variables and 470 observations and is composed of numeric, factor, 
# and logical variables. All variables have uneven and skewed distributions, but there are no
# missing observations.



# 4. data cleaning
#correlations
correlations <- cor(xTrainNum)
par(mfrow = c(1,1))
corrplot(correlations, method = "number", order = "hclust")
# There are no highly correlated variables.  Most variables have very little or no correlation.



#5. Model implementation of algorithms (with standardized cleaning) and with built-in feature selection
# Set up dummy variables to train on some of the models
dmy <- dummyVars(" ~ .", data = xTrain, fullRank = T)
xTrainDummy <- data.frame(predict(dmy, newdata = xTrain))

xTrainNum <- sapply(xTrain[,1:16], as.numeric)

# Controlled resampling
set.seed(seed)
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                     summaryFunction = defaultSummary, sampling = "smote", 
                     classProbs = TRUE, savePredictions = "final")

#nearest shrunken centroid
gridNSC <- expand.grid(threshold = c(0,1))
set.seed(seed)
modelNSC <- train(x = xTrainDummy, y = yTrain, 
                  method = "pam",
                  preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv"),
                  tuneGrid = gridNSC,
                  trControl = ctrl,
                  metric = "Kappa")
modelNSC # Accuracy = 0.760, Kappa = 0.134
beep(2)

# Partial Least Squares Discriminant Analysis (WAIT TIME ~ 10-15 mins)
gridPLS = expand.grid(mtry = c(20, 30))
set.seed(seed)
modelPLS <- train(x = xTrainDummy, y = yTrain, 
                  method = "ORFpls",
                  tuneGrid = gridPLS,
                  trControl = ctrl,
                  preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv"),
                  metric = "Kappa")
modelPLS # Accuracy = .763, Kappa = .07
beep(3)

# Penalized Logistic Regression (no built-in feature selection)
gridPLR = expand.grid(lambda = seq(0.8, 1.3, by = 0.1), cp = c('aic', 'bic'))
set.seed(seed)
modelPLR <- train(x = xTrainDummy, y = yTrain, 
                  method = "plr",
                  tuneGrid = gridPLR,
                  preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                  metric = "Kappa",
                  trControl = ctrl)
modelPLR # Accuracy = 0.755, Kappa = 0.148
beep(2)

# Penalized Linear Discriminant Analysis
gridPLDA = expand.grid(lambda = seq(0.1, 0.2, by = 0.01), K = 1)
set.seed(seed)
modelPLDA <- train(x = xTrainDummy, y = yTrain, 
                   method = "PenalizedLDA",
                   tuneGrid = gridPLDA,
                   preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv"),
                   metric = "Kappa",
                   trControl = ctrl)
modelPLDA # Accuracy = 0.729, Kappa = 0.177
beep(2)

# support vector machines (no built-in feature selection)
library(kernlab)
gridSVM <- expand.grid(sigma = seq(0.3, 0.7, by = 0.1), C = seq(0.3, 0.7, by = 0.1))
set.seed(seed)
modelSVM <- train(x = xTrainNum, 
                  y = yTrain,
                  method = "svmRadial",
                  tuneGrid = gridSVM,
                  preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv"),
                  metric = "Kappa",
                  fit = FALSE,
                  trControl = ctrl)
modelSVM
beep(2) # Accuracy = 0.649, Kappa = 0.125

#random forest
library(randomForest)
gridRF <- expand.grid(mtry = seq(30, 40, by = 1))
set.seed(seed)
ModelRF <- train(x = xTrainNum, y = yTrain, 
                 method = "rf",
                 tuneGrid = gridRF,
                 preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv"),
                 metric = "Kappa",
                 trControl = ctrl)
ModelRF# Accuracy = .732, Kappa = .145
beep(2)

#flexble discriminant analysis
library(mda)
modelFDA <- fda(PRE4 ~., data = xTrainNum)
modelFDA

#bagged trees
set.seed(seed)
ModelBT <- train(x = xTrain, y = yTrain,
                         method = "treebag",
                         preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                         nbagg = 50,  
                         trControl = ctrl)

ModelBT #Accuracy = .74, Kappa = .18
