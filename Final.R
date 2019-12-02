#Final Exam - Group 4
#George Garcia, Marco Duran Perez, Hilary Balli

library(readr)
final <- read.csv("C:/Users/Hilary/Desktop/QUMT 6350/examdata.csv")
head(final)
str(final)

library(caret)
library(AppliedPredictiveModeling)
library(corrplot)
library(skimr)
library(dplyr)

summary(final)
dataFinal <- final

x <- subset(dataFinal, select = -Risk1Yr)
y <- subset(dataFinal, select = Risk1Yr)

#review imbalances in outcome variable
barplot(table(y), 
        names.arg = c("Died", "Survived"),
        col = c("blue", "green"),
        main = "Risk1Yr")

#set seet
seed <- 250

#set training and testing data
dataFinal <- na.omit(dataFinal)
x <- subset(dataFinal, select = -Risk1Yr)
y <- subset(dataFinal, select = Risk1Yr)
Ftrain <- createDataPartition(y$Risk1Yr,
                              p = .75,
                              list = FALSE)
xTest <- x[-Ftrain,]
yTest <- y[-Ftrain,]

xTrain <- x[Ftrain,]
yTrain <- y[Ftrain,]

#explore data
#skewed results
skimmed1 <- skim_to_wide(xTrain)
skimmed2 <- skim_to_wide(yTrain)
View(skimmed1)
View(skimmed2)

#set x variables to numeric
xTrain[,1:16] <- sapply(xTrain[,1:16],as.numeric)
xTrainNum <- xTrain[,1:16]
str(xTrainNum)

#boxplot of predictors
boxplot(xTrainNum)

# Histograms of predictors and outcome variables
par(mfrow = c(3,3))
hist(xTrainNum$ï..Attribute)
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

#data cleaning
#correlations
correlations <- cor(xTrainNum)
par(mfrow = c(1,1))
corrplot(correlations, method = "number", order = "hclust")

#5. Model implementation of algorithms (with standardized cleaning)

# Controlled resampling
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                     summaryFunction = defaultSummary, sampling = "smote", 
                     classProbs = TRUE, savePredictions = "final")

#nearest shrunken centroid
gridNSC <- expand.grid(threshold = seq(0.5, 2.5, by = 0.1))
set.seed(seed)
modelNSC <- caret::train(x = xTrain, y = yTrain, 
                         method = "pam",
                         preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                         tuneGrid = gridNSC,
                         trControl = ctrl)
modelNSC # Accuracy = .85, Kappa = 0.0


# Partial Least Squares Discriminant Analysis
gridPLS = expand.grid(ncomp = 1:5)
set.seed(seed)
modelPLS <- train(x = xTrainNum, y = yTrain, 
                  method = "pls",
                  tuneGrid = gridPLS,
                  trControl = ctrl,
                  preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"))
modelPLS # Accuracy = .67, Kappa = .053

# Penalized Logistic Regression
gridPLR = expand.grid(lambda = seq(0.015, 0.030, by = 0.001),
                      cp = 'bic')
set.seed(seed)
modelPLR <- train(x = xTrainNum, y = yTrain, 
                  method = "plr",
                  tuneGrid = gridPLR,
                  preProcess = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                  trControl = ctrl)
modelPLR # 


# Penalized Linear Discriminant Analysis
gridPLDA = expand.grid(diagonal = c(TRUE, FALSE), lambda = seq(0.015, 0.030, by = 0.001))
set.seed(seed)
modelPLDA <- train(x = xTrainNum, y = yTrain, 
                   method = "sda",
                   tuneGrid = gridPLDA,
                   preProcess = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                   trControl = ctrl)
modelPLDA # Accuracy = .67, Kappa = .03

# support vector machines
library(kernlab)
set.seed(seed)
modelSVM <- caret::train(x = xTrainNum, 
                         y = yTrain,
                         method = "svmRadial",
                         preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                         fit = FALSE,
                         trControl = ctrl)
modelSVM #Accuracy = .72, Kappa = .05

#random forest
library(randomForest)
gridRF <- expand.grid(mtry = seq(0.5, 2, by = 0.5))
set.seed(seed)
ModelRF <- caret::train(x = xTrainNum, y = yTrain, 
                              method = "rf", 
                              preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                              tuneGrid = gridRF,
                              trControl = ctrl)
ModelRF # Accuracy = .75, Kappa = .14

#flexble discriminant analysis
library(mda)
modelFDA <- fda(PRE4 ~., data = xTrainNum)
modelFDA

#bagged trees
set.seed(seed)
ModelBT <- caret::train(x = xTrain, y = yTrain,
                         method = "treebag",
                         preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                         nbagg = 50,  
                         trControl = ctrl)

ModelBT #Accuracy = .74, Kappa = .18
