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
final <- read.csv("C:/Users/Hilary/Desktop/QUMT 6350/examdata.csv")



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
modelSVM <- train(x = xTrainDummy, 
                  y = yTrain,
                  method = "svmRadial",
                  tuneGrid = gridSVM,
                  preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv"),
                  metric = "Kappa",
                  fit = FALSE,
                  trControl = ctrl)
modelSVM #Accuracy = .713, Kappa = -.0125 -Num
beep(2) # Accuracy = 0.54, Kappa = 0.108 - Dummy

#random forest
library(randomForest)
gridRF <- expand.grid(mtry = seq(30, 40, by = 1))
set.seed(seed)
ModelRF <- train(x = xTrainDummy, y = yTrain, 
                 method = "rf",
                 tuneGrid = gridRF,
                 preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv"),
                 metric = "Kappa",
                 trControl = ctrl)
ModelRF# Accuracy = .732, Kappa = .145 -Num
#accuracy = .76 , Kappa = .141 -Dummy
beep(2)

#flexble discriminant analysis
library(mda)
modelFDA <- fda(PRE4 ~., data = xTrainDummy)
modelFDA #training misclassification error = .74

#bagged trees
set.seed(seed)
ModelBT <- train(x = xTrainDummy, y = yTrain,
                 method = "treebag",
                 preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"),
                 nbagg = 50,  
                 trControl = ctrl)

ModelBT #Accuracy = .75, Kappa = .15

#C5.0
library(C50)
set.seed(seed)
modelC5 <- caret::train(x = xTrainDummy,
                        y = yTrain,
                        method = "C5.0",
                        preProc = c("center", "scale"),
                        verbose = FALSE,
                        trControl = ctrl)
modelC5 # Accuracy = .79, Kappa = .124

# eXtreme Gradient Boosted Tree
library(xgboost)
set.seed(seed)
modelXGBT <- caret::train(x = xTrainDummy, y = yTrain, 
                          method = "xgbTree",
                          preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"), 
                          trControl = ctrl)
modelXGBT # Accuracy =.74 , Kappa =.10


# eXtreme Gradient Boosted DART
set.seed(seed)
modelXGBD <- caret::train(x = xTrainDummy, y = yTrain, 
                          method = "xgbDART",
                          preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"), 
                          trControl = ctrl)
modelXGBD # Accuracy = , Kappa = 


# eXtreme Gradient Boosted Linear
set.seed(seed)
modelXGBL <- caret::train(x = xTrainDummy, y = yTrain, 
                          method = "xgbLinear",
                          preProc = c("center", "scale", "YeoJohnson", "spatialSign", "zv", "nzv"), 
                          trControl = ctrl)
modelXGBL # Accuracy = .75 , Kappa = .10

#Statistical Analysis

# Resampled model results
modelResults <- resamples(list(PLS = modelPLS,
                               PLR = modelPLR, PLDA = modelPLDA, NSC = modelNSC,
                               SVM = modelSVM, BT = ModelBT, RF = ModelRF, C5 = modelC5,
                               C5 = modelC5, XGBT = modelXGBT, XGBL = modelXGBL))
summary(modelResults)
dotplot(modelResults)

# Check differences
modelDiff <- diff(modelResults)


#Accuracy: Testing Data

# Prepare training data
preProcParams <- preProcess(xTrainNum, method = c("center", "scale"))
xTrainTrans <- predict(preProcParams, xTrainNum)

# Train final model
finalModel <-  C5.0(x = xTrainTrans, y = yTrain, trials = 50)
finalModel
summary(finalModel)

# Prepare testing data
xTestNum <- as.data.frame(lapply(xTest, function(x) as.numeric(as.character(x))))
set.seed(seed)
xTestTrans <- predict(preProcParams, xTestNum)

# Test data predictions
predictions <- predict(finalModel, newdata = xTestTrans, neighbors = 3)

# Determine accuracy
confusionMatrix(predictions, yTest)

# Accuracy on testing data is 0.85, with a sensitivity = 0.0 and specificity = 1.0.

#caretEnsemble

# Train a list of models using caretList()
algoList <- c('C5.0', 'svmRadial', 'sda')
models <- caretList(x = xTrainNum, y = yTrain,
                    methodList = algoList,
                    tuneList = list(
                      rf1 = caretModelSpec(method = "rf", tuneGrid = data.frame(.mtry=2)),
                      rf2 = caretModelSpec(method = "rf", tuneGrid = data.frame(.mtry=10), preProcess = "pca"),
                      nn = caretModelSpec(method = "nnet", trace = FALSE, tuneLength = 2)
                    ),
                    preProc = c("center", "scale"),
                    trControl = ctrl)

# Combine with caretEnsemble()
ensemble1 <- caretEnsemble(models, metric = "Accuracy", trControl = trainControl(number = 6))
summary(ensemble1)

# caretEnsemble resulted in a combined accuracy of 0.843.

# Combine outputs with caretStack()
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3,
                             savePredictions=TRUE)
set.seed(seed)
stackGLM <- caretStack(models, method="glm",  
                       preProc=c("center", "scale"), 
                       trControl=stackControl)
stackGLM # Accuracy = 0.85


set.seed(seed)
stackRF <- caretStack(models, method="rf",
                      preProc=c("center", "scale"), 
                      trControl=stackControl)
stackRF # Accuracy = error

#DALEX
library(DALEX)
library(gbm)
set.seed(seed)
regr_rf <- caret::train(i..Attributes~., data = xTrainNum, method = "rf", ntree = 100)

regr_gbm <- caret::train(i..Attributes~. , data = xTrainNum, method = "gbm")

regr_nn <- caret::train(i..Attributes~., data = xTrainNum,
                        method = "nnet",
                        linout = TRUE,
                        preProcess = c('center', 'scale'),
                        maxit = 500,
                        tuneGrid = expand.grid(size = 2, decay = 0),
                        trControl = trainControl(method = "none", seeds = 1))

### Explain Function ###

explainer_regr_rf <- DALEX::explain(regr_rf, label = "rf", 
                                    data = xTrainNum, y = as.numeric(y$Risk1Yr),
                                    colorize = FALSE)

explainer_regr_gbm <- DALEX::explain(regr_gbm, label = "gbm", 
                                     data = xTrainNum, y = as.numeric(y$Risk1Yr),
                                     colorize = FALSE)

explainer_regr_nn <- DALEX::explain(regr_nn, label = "nn", 
                                    data = xTrainNum, y = as.numeric(y$Risk1Yr),
                                    colorize = FALSE)
### DALEX Model Performance ###

mp_regr_rf <- model_performance(explainer_regr_rf)
mp_regr_gbm <- model_performance(explainer_regr_gbm)
mp_regr_nn <- model_performance(explainer_regr_nn)

mp_regr_rf
mp_regr_gbm
mp_regr_nn

plot(mp_regr_rf, mp_regr_nn, mp_regr_gbm)
plot(mp_regr_rf, mp_regr_nn, mp_regr_gbm, geom = "boxplot")

### DALEX Variable Importance ###

vi_regr_rf <- variable_importance(explainer_regr_rf, loss_function = loss_root_mean_square)
vi_regr_gbm <- variable_importance(explainer_regr_gbm, loss_function = loss_root_mean_square)
vi_regr_nn <- variable_importance(explainer_regr_nn, loss_function = loss_root_mean_square)

plot(vi_regr_rf, vi_regr_gbm, vi_regr_nn)

### iml - Partial Dependence Plot ###
library(iml)
library(mlr)
X = xTrainNum[which(names(xTrainNum) != "Risk1Yr")]
predictor <- Predictor$new(ModelRF, data = x, y = dataFinal[,16])
str(predictor)

library(dplyr)
pdp_obj <- Partial$new(predictor, feature = "AGE")
pdp_obj$center(min(xTrainNum$AGE))
glimpse(pdp_obj$results)

pdp_obj$plot()

### ICE plots ###

pdp_obj2 <- Partial$new(predictor, feature = c("PRE7", "PRE8"))
pdp_obj2$plot()

### Tree Surrogate ###

tree <- TreeSurrogate$new(predictor, maxdepth = 5)
tree$r.squared
plot(tree)
tree$results %>%
  mutate(prediction = colnames(select(., .y.hat.qual_high, .y.hat.qual_low))[max.col(select(., .y.hat.qual_high, .y.hat.qual_low),
                                                                                     ties.method = "first")],
         prediction = ifelse(prediction == "???", "???", "???")) %>%
  ggplot(aes(x = predictor, fill = prediction)) +
  facet_wrap(~ .path, ncol = 5) +
  geom_bar(alpha = 0.8) +
  scale_fill_tableau() +
  guides(fill = FALSE)

### Local Model - Local Interpretable Model-agnostic Explanations ###

X2 <- xTest[, -16]
i = 1
lime_explain <- LocalModel$new(predictor, x.interest = X2[i, ])
lime_explain$results

### plot(lime_explain) ###

p1 <- lime_explain$results %>%
  ggplot(aes(x = reorder(feature.value, -effect), y = effect, fill = .class)) +
  facet_wrap(~ .class, ncol = 2) +
  geom_bar(stat = "identity", alpha = 0.8) +
  coord_flip() +
  labs(title = paste0("Test case #", i)) +
  guides(fill = FALSE)

### Shapley Value ###
shapley <- Shapley$new(predictor, x.interest = X2[1, ])
head(shapley$results)

### shapley$plot() ###
shapley$results %>%
  ggplot(aes(x = reorder(feature.value, -phi), y = phi, fill = class)) +
  facet_wrap(~ class, ncol = 2) +
  geom_bar(stat = "identity", alpha = 0.8) +
  coord_flip() +
  guides(fill = FALSE)






### Neural Networks: Example with Categorical Response at Two Levels ###

## Min-Max Normalization ##
dataFinal$PRE4 <- (dataFinal$PRE4 - min(dataFinal$PRE4))/(max(dataFinal$PRE4) - min(dataFinal$PRE4))
dataFinal$PRE5 <- (dataFinal$PRE5 - min(dataFinal$PRE5))/(max(dataFinal$PRE5) - min(dataFinal$PRE5))
dataFinal$PRE7 <- (dataFinal$PRE7 - min(dataFinal$PRE7))/(max(dataFinal$PRE7) - min(dataFinal$PRE7))
dataFinal$PRE8 <- (dataFinal$PRE8 - min(dataFinal$PRE8))/(max(dataFinal$PRE8) - min(dataFinal$PRE8))
dataFinal$PRE9 <- (dataFinal$PRE9 - min(dataFinal$PRE9))/(max(dataFinal$PRE9) - min(dataFinal$PRE9))
dataFinal$PRE10 <- (dataFinal$PRE10 - min(dataFinal$PRE10))/(max(dataFinal$PRE10) - min(dataFinal$PRE10))
dataFinal$PRE11 <- (dataFinal$PRE11 - min(dataFinal$PRE11))/(max(dataFinal$PRE11) - min(dataFinal$PRE11))
dataFinal$PRE17 <- (dataFinal$PRE17 - min(dataFinal$PRE17))/(max(dataFinal$PRE17) - min(dataFinal$PRE17))
dataFinal$PRE19 <- (dataFinal$PRE19 - min(dataFinal$PRE19))/(max(dataFinal$PRE19) - min(dataFinal$PRE19))
dataFinal$PRE25 <- (dataFinal$PRE25 - min(dataFinal$PRE25))/(max(dataFinal$PRE25) - min(dataFinal$PRE25))
dataFinal$PRE30 <- (dataFinal$PRE30 - min(dataFinal$PRE30))/(max(dataFinal$PRE30) - min(dataFinal$PRE30))
dataFinal$PRE32 <- (dataFinal$PRE32 - min(dataFinal$PRE32))/(max(dataFinal$PRE32) - min(dataFinal$PRE32))
dataFinal$AGE <- (dataFinal$AGE - min(dataFinal$AGE))/(max(dataFinal$AGE) - min(dataFinal$AGE))

## Data Partition ##
set.seed(seed)
ind <- sample(2,nrow(dataFinal), replace = TRUE, prob = c(0.8, 0.2))
training <- dataFinal[ind==1,]
testing <- dataFinal[ind==2,]

## Neural Networks ##
library(neuralnet)
set.seed(seed)
n <- neuralnet(Risk1Yr~AGE+PRE4+PRE5+PRE7+PRE8+PRE9+PRE10+PRE11+PRE17+PRE19+PRE25+PRE30+PRE32,
               data = training,
               hidden = 1,
               err.fct = "ce",
               linear.output = FALSE,
               lifesign = 'full',
               algorithm = "rprop+"
)
plot(n)

## Prediction ##

output <- compute(n, training[,-17])
output
head(output$net.result)
head(training[1,])

## Node Output Calculations with Sigmoid Activation Funtion ##
in14 <- -1.24168 + (-1.18182*0.4032922) + (32.88539*0.01078041) + (-1.24769*0) + (0.02539*0) + (-28.12965*0) + (-0.15854*0) + (-0.37519*0) + (-1.83694*0) + (0.57395*0) + (66.73049*0) + (-0.66428*1) + (38.72126*0) + (-0.03061*.4545455)
in14

out14 <- 1/(1+exp(-in14))
out14

in15 <- -0.57596 + (-10.19008*out14)
in15

out15 <- 1/(1+exp(-in15))
out15

in16 <- 0.57627 + (10.18588*out14)
in16

out16 <- 1/(1+exp(-in16))
out16

## Confusion Matrix & Misclassication Error - training data ##
output <- compute(n, training[,-17])
p1 <- output$net.result
pred1 <- ifelse(p1>0.5, 1, 0)
tab1 <- table(pred1, training$Risk1Yr)
tab1

## Confusion Matrix & Misclassication Error - testing data ##
output2 <- compute(n, training[,-17])
p2 <- output$net.result
pred2 <- ifelse(p2>0.5, 1, 0)
tab2 <- table(pred2, training$Risk1Yr)
tab2
