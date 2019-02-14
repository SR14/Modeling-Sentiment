### Directory ###
getwd()
setwd("/Users/sergiorobledo/Desktop/")

### Required Packages ###
install.packages("doParallel")
library(doParallel)
install.packages("caret")
library(caret)
install.packages("plotly")
library(plotly)
install.packages("corrplot")
library(corrplot)
library(e1071)
install.packages("kknn")
library(kknn)
library(dplyr)
library(randomForest)
library(C50)

### Parallel Processing ###
detectCores()
cl <- makeCluster(2)
registerDoParallel(cl)
getDoParWorkers()
stopCluster(cl)

###                 ###
### iPhone Workflow ###
###                 ###

### Explore the Data ###
iPhone <- read.csv("iphone_smallmatrix_labeled_8d.csv")
str(iPhone)
summary(iPhone)
plot_ly(iPhone, x= ~iPhone$iphonesentiment, type='histogram')
apply(is.na(iPhone),2,sum)
iPhone$iphonesentiment <- as.factor(iPhone$iphonesentiment)

### Preprocessing & Feature Selection ###
options(max.print = 1000000)
corrData <- cor(iPhone)
corrplot(corrData)
print(corrData)
iPhoneCOR <- iPhone
iPhoneCOR$samsunggalaxy <- NULL
iPhoneCOR$htcphone <- NULL
nzvMetrics <- nearZeroVar(iPhoneCOR, saveMetrics = TRUE)
nzvMetrics
nzv <- nearZeroVar(iPhoneCOR, saveMetrics = FALSE)
nzv
iPhoneNZV <- iPhoneCOR[,-nzv]
iPhoneNZV$iphonesentiment <- as.factor(iPhoneNZV$iphonesentiment)
iPhoneNZV$iphone <- NULL
str(iPhoneNZV)

# Using Recursive Feature Elimination on orginal DF 
set.seed(123)
iPhoneSample <- iPhone[sample(1:nrow(iPhone),1000,replace = FALSE),]
ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)
rfeResults <- rfe(iPhoneSample[,1:58],
                  iPhoneSample$iphonesentiment,
                  sizes = (1:58),
                  rfeControl = ctrl)
rfeResults$optVariables
plot(rfeResults,type = c("g","o"))
iPhoneRFE <- iPhone[,predictors(rfeResults)]
iPhoneRFE$iphonesentiment <- as.factor(iPhone$iphonesentiment)
str(iPhoneRFE)

### Model Development and Evaluation ###
set.seed(123)
inTraining <- createDataPartition(iPhone$iphonesentiment, p=0.70, list = FALSE)
trainSet <- iPhone[inTraining,]
testSet <- iPhone[-inTraining,]
str(trainSet)
str(testSet)

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

# C5 #
set.seed(123)
C5_Fit <- C50::C5.0(iphonesentiment~.,data = trainSet, method = "C5.0", 
                    trControl = fitControl, tuneLength=2, rules = TRUE)
C5_Fit
varImp(C5_Fit)

C5_Prediction <- predict(C5_Fit,iPhone)
postResample(C5_Prediction,testSet$iphonesentiment)
summary(C5_Prediction)
C5_Prediction

# Random Forest (RF) #
set.seed(123)
rdGrid <- expand.grid(mtry = c(2,4,8,16,32))
RF_Fit <- train(iphonesentiment~.,data = trainSet, 
                method = "rf", importance = T, 
                trControl = fitControl,tuneGrid = rdGrid)
RF_Fit
varImp(RF_Fit)

RF_Prediction <- predict(RF_Fit,iPhone)
postResample(RF_Prediction,testSet$iphonesentiment)
summary(RF_Prediction)


# Support Vector Machine (SVM) #
set.seed(123)
SVM_Fit <- svm(iphonesentiment~., data = trainSet)
SVM_Fit

SVM_Prediction <- predict(SVM_Fit, iPhone)
postResample(SVM_Prediction,testSet$iphonesentiment)
summary(SVM_Prediction)

# Weighted K-Nearest Neighbor #
set.seed(123)
KKNN_Fit <- train.kknn(iphonesentiment~., data = trainSet)
KKNN_Fit

KKNN_Prediction <- predict(KKNN_Fit, iPhone)
postResample(KKNN_Prediction, testSet$iphonesentiment)
summary(KKNN_Prediction)

### Model Using Your Feature Selection Data Sets ###
set.seed(123)
inTrainingNZV <- createDataPartition(iPhoneNZV$iphonesentiment, 
                                     p=0.70, list = FALSE)
trainSetNZV <- iPhoneNZV[inTrainingNZV,]
testSetNZV <- iPhoneNZV[-inTrainingNZV,]
str(trainSetNZV)
str(testSetNZV)

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

# C5 NZV #
set.seed(123)
C5_FitNZV <- C50::C5.0(iphonesentiment~.,data = trainSetNZV, 
                   method = "C5.0", trControl = fitControl,
                   tuneLength=2, rules = TRUE)
C5_FitNZV
varImp(C5_FitNZV)

C5_PredictionNZV <- predict(C5_FitNZV,testSetNZV)
postResample(C5_PredictionNZV,testSetNZV$iphonesentiment)
summary(C5_PredictionNZV)
C5_PredictionNZV
cmC5 <- confusionMatrix(C5_PredictionNZV, testSetNZV$iphonesentiment)
cmC5

# C5 COR #
set.seed(123)
inTrainingCOR <- createDataPartition(iPhoneCOR$iphonesentiment,
                                     p=0.70, list = FALSE)
trainSetCOR <- iPhoneCOR[inTrainingCOR,]
testSetCOR <- iPhoneCOR[-inTrainingCOR,]
str(trainSetCOR)
str(testSetCOR)
set.seed(123)
C5_Fit_COR <- C50::C5.0(iphonesentiment~.,data = trainSetCOR, 
                        method = "C5.0", trControl = fitControl,
                        tuneLength=2, rules = TRUE)
C5_Prediction_COR <- predict(C5_Fit_COR,testSetCOR)
postResample(C5_Prediction_COR,testSetCOR$iphonesentiment)
summary(C5_Prediction_COR)
cmC5_COR <- confusionMatrix(C5_Prediction_COR, testSetCOR$iphonesentiment)
cmC5_COR

# C5 RFE #
set.seed(123)
inTrainingRFE <- createDataPartition(iPhoneRFE$iphonesentiment, 
                                     p=0.70, list = FALSE)
trainSetRFE <- iPhoneRFE[inTrainingRFE,]
trainSetRFE$iphonesentiment <- as.factor(trainSetRFE$iphonesentiment)
testSetRFE <- iPhoneRFE[-inTrainingRFE,]
testSetRFE$iphonesentiment <- as.factor(testSetRFE$iphonesentiment)
str(trainSetRFE)
str(testSetRFE)
set.seed(123)
C5_Fit_RFE <- C50::C5.0(iphonesentiment~.,data = trainSetRFE, 
                        method = "C5.0", trControl = fitControl,
                        tuneLength=2, rules = TRUE)
C5_Prediction_RFE <- predict(C5_Fit_RFE,testSetRFE)
postResample(C5_Prediction_RFE,testSetRFE$iphonesentiment)
summary(C5_Prediction_COR)
cmC5_RFE <- confusionMatrix(C5_Prediction_RFE, testSetRFE$iphonesentiment)
cmC5_RFE

# Random Forest (RF) #
set.seed(123)
rdGrid <- expand.grid(mtry = c(2,4,8,16,32))
RF_FitNZV <- train(iphonesentiment~.,data = trainSetNZV, method = "rf",
                   importance = T, trControl = fitControl,tuneGrid = rdGrid)
RF_FitNZV
varImp(RF_FitNZV)

RF_PredictionNZV <- predict(RF_FitNZV,testSetNZV)
postResample(RF_PredictionNZV,testSetNZV$iphonesentiment)
summary(RF_PredictionNZV)
cmRF <- confusionMatrix(RF_PredictionNZV, testSetNZV$iphonesentiment)
cmRF

# Support Vector Machine (SVM) #
set.seed(123)
SVM_FitNZV <- svm(iphonesentiment~., data = trainSetNZV, kernel = "polynomial")
SVM_FitNZV

SVM_PredictionNZV <- predict(SVM_FitNZV, testSetNZV)
postResample(SVM_PredictionNZV,testSetNZV$iphonesentiment)
summary(SVM_PredictionNZV)
cmSVM <- confusionMatrix(SVM_PredictionNZV,testSetNZV$iphonesentiment)
cmSVM

### Feature Engineering & Modeling ###

# recode() function #
iPhoneRC <- iPhoneRFE
iPhoneRC$iphonesentiment <- as.factor(recode(iPhoneRFE$iphonesentiment,
                                             '0' = 1, '1' = 1, '2' = 2, 
                                             '3' = 3, '4' = 4, '5' = 4))
summary(iPhoneRC)
str(iPhoneRC)

set.seed(123)
inTrainingRC <- createDataPartition(iPhoneRC$iphonesentiment, 
                                    p = 0.70, list = FALSE)
trainSetRC <- iPhoneRC[inTrainingRC,]
testSetRC <- iPhoneRC[-inTrainingRC,]

set.seed(123)
C5_Fit_RC <- C50::C5.0(iphonesentiment~., data = trainSetRC,
                   method = "C5.0", trControl = fitControl,
                   tuneLength = 2, rules = TRUE)
C5_Prediction_RC <- predict(C5_Fit_RC, testSetRC)
postResample(C5_Prediction_RC, testSetRC$iphonesentiment)
summary(C5_Prediction_RC)
cmC5_RC <- confusionMatrix(C5_Prediction_RC, testSetRC$iphonesentiment)
cmC5_RC

# principal component analysis #
iPhonePCA <- iPhone
preprocessParams <- preProcess(trainSet[,-59], 
                               method = c("center","scale","pca"), 
                               thresh = 0.95)
train_pca <- predict(preprocessParams, trainSet[,-59])
train_pca$iphonesentiment <- trainSet$iphonesentiment
test_pca <- predict(preprocessParams, testSet[,-59])
test_pca$iphonesentiment <- testSet$iphonesentiment
print(preprocessParams)
str(train_pca)
str(test_pca)

set.seed(123)
C5_Fit_PCA <- C50::C5.0(iphonesentiment~., data = train_pca,
                        method = "C5.0", trControl = fitControl,
                        tuneLength = 2, rules = TRUE)
C5_Prediction_PCA <- predict(C5_Fit_PCA, test_pca)
postResample(C5_Prediction_PCA, test_pca$iphonesentiment)
cmC5_PCA <- confusionMatrix(C5_Prediction_PCA, test_pca$iphonesentiment)
cmC5_PCA

### Best Model: C5.0, Best Feature Selection Method: RFE with Recode() ###
### Apply Model to Data ###
# Import iPhone Large Matrix
iPhoneLarge <- read.csv("iPhoneLargeMatrix.csv")
iPhoneLarge$iphonesentiment <- as.factor(iPhoneLarge$iphonesentiment)
str(iPhoneLarge)
apply(is.na(iPhoneLarge),2,sum)

# Recursive Feature Elimination & Recoding
set.seed(123)
rfeResults$optVariables
plot(rfeResults,type = c("g","o"))
iPhoneLargeRFE <- iPhoneLarge[,predictors(rfeResults)]
iPhoneLargeRFE$iphonesentiment <- iPhoneLarge$iphonesentiment
str(iPhoneLargeRFE)

# Prediction 
set.seed(123)
C5_Prediction_RFE_Large <- predict(C5_Fit_RC,iPhoneLargeRFE)
summary(C5_Prediction_RFE_Large)
C5_Prediction_RFE_Large
write.csv(C5_Prediction_RFE_Large, "iPhoneLargePredictino.csv")
str(iPhoneLargeRFE)

###                 ###
### Galaxy Workflow ###
###                 ###

### Explore Data ###
Galaxy <- read.csv("galaxy_smallmatrix_labeled_9d.csv")
Galaxy$galaxysentiment <- as.factor(Galaxy$galaxysentiment)
plot_ly(Galaxy, x= ~Galaxy$galaxysentiment, type='histogram')
apply(is.na(Galaxy),2,sum)
summary(Galaxy)
str(Galaxy)

### Model Selection ###

# Training & Testing Sets
set.seed(123)
Galaxy_Training <- createDataPartition(Galaxy$galaxysentiment,
                                       p=0.70, list = FALSE)
trainGalaxy <- Galaxy[Galaxy_Training,]
testGalaxy <- Galaxy[-Galaxy_Training,]
str(trainGalaxy)
str(testGalaxy)
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

# C5.0
set.seed(123)
C5_Fit_Galaxy <- C50::C5.0(galaxysentiment~.,data = trainGalaxy, 
                           method = "C5.0", 
                           trControl = fitControl, 
                           tuneLength=2, 
                           rules = TRUE)
C5_Prediction_Galaxy <- predict(C5_Fit_Galaxy, testGalaxy)
postResample(C5_Prediction_Galaxy,testGalaxy$galaxysentiment)
summary(C5_Prediction_Galaxy)
C5_Prediction_Galaxy

# Random Forest (RF) #
set.seed(123)
rdGrid <- expand.grid(mtry = c(2,4,8,16,32))
RF_Fit_Galaxy <- randomForest(galaxysentiment~.,data = trainGalaxy, 
                       method = "rf", importance = T,
                       trControl = fitControl, tuneGrid = rdGrid)
RF_Prediction_Galaxy <- predict(RF_Fit_Galaxy, testGalaxy)
postResample(RF_Prediction_Galaxy,testGalaxy$galaxysentiment)
summary(RF_Prediction_Galaxy)
RF_Prediction_Galaxy

# Support Vector Machines (SVM)
set.seed(123)
SVM_Fit_Galaxy <- svm(galaxysentiment~., data = trainGalaxy)
SVM_Prediction_Galaxy <- predict(SVM_Fit_Galaxy, testGalaxy)
postResample(SVM_Prediction_Galaxy, testGalaxy$galaxysentiment)
summary(SVM_Prediction_Galaxy)
SVM_Prediction_Galaxy

# KKNN
set.seed(123)
KKNN_Fit_Galaxy <- train.kknn(galaxysentiment~., data = trainGalaxy)
KKNN_Prediction_Galaxy <- predict(KKNN_Fit_Galaxy, testGalaxy)
postResample(KKNN_Prediction_Galaxy, testGalaxy$galaxysentiment)
KKNN_Prediction_Galaxy

# C5 is the Best Model 

### Feature Selection ###

# COR
options(max.print = 10000)
corrGalaxy <- cor(Galaxy)
print(corrGalaxy)

# NZV 
nzvMetricsGalaxy <- nearZeroVar(Galaxy, saveMetrics = TRUE)
nzvMetricsGalaxy
nzvGalaxy <- nearZeroVar(Galaxy, saveMetrics = FALSE)
nzvGalaxy
GalaxyNZV <- Galaxy[,-nzvGalaxy]
GalaxyNZV$galaxysentiment <- as.factor(GalaxyNZV$galaxysentiment)
str(GalaxyNZV)

# RFE 
set.seed(123)
GalaxySample <- Galaxy[sample(1:nrow(Galaxy),1000,replace = FALSE),]
ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)
rfeResultsGalaxy <- rfe(GalaxySample[,1:58],
                        GalaxySample$galaxysentiment,
                        sizes = (1:58),
                        rfeControl = ctrl)
rfeResultsGalaxy$optVariables
plot(rfeResults,type = c("g","o"))
GalaxyRFE <- Galaxy[,predictors(rfeResultsGalaxy)]
GalaxyRFE$galaxysentiment <- as.factor(Galaxy$galaxysentiment)
str(GalaxyRFE)

# PCA
trainGalaxyPCA <- predict(preprocessParams, trainGalaxy[,-59])
trainGalaxyPCA$galaxysentiment <- trainGalaxy$galaxysentiment
testGalaxyPCA <- predict(preprocessParams, testGalaxy[,-59])
testGalaxyPCA$galaxysentiment <- testGalaxy$galaxysentiment
str(trainGalaxyPCA)
str(testGalaxyPCA)

# NZV Dataset Modeling w/ C5
set.seed(123)
GalaxyTrainingNZV <- createDataPartition(GalaxyNZV$galaxysentiment,
                                         p=0.70, list = FALSE)
trainGalaxyNZV <- GalaxyNZV[GalaxyTrainingNZV,]
testGalaxyNZV <- GalaxyNZV[-GalaxyTrainingNZV,]
set.seed(123)
C5_Fit_GalaxyNZV <- C5.0(galaxysentiment~.,data = trainGalaxyNZV, 
                           method = "C5.0", 
                           trControl = fitControl, 
                           tuneLength=2, 
                           rules = TRUE)
C5_Prediction_GalaxyNZV <- predict(C5_Fit_GalaxyNZV, testGalaxyNZV)
postResample(C5_Prediction_GalaxyNZV, testGalaxyNZV$galaxysentiment)
summary(C5_Prediction_GalaxyNZV)

# RFE Dataset Modeling w/ C5
set.seed(123)
GalaxyTrainingRFE <- createDataPartition(GalaxyRFE$galaxysentiment,
                                         p = 0.70, list = FALSE)
trainGalaxyRFE <- GalaxyRFE[GalaxyTrainingRFE,]
testGalaxyRFE <- GalaxyRFE[-GalaxyTrainingRFE,]
set.seed(123)
C5_Fit_Galaxy_RFE <- C5.0(galaxysentiment~., data = trainGalaxyRFE,
                          rules = TRUE)
C5_Prediction_Galaxy_RFE <- predict(C5_Fit_Galaxy_RFE, testGalaxyRFE)
postResample(C5_Prediction_Galaxy_RFE, testGalaxyRFE$galaxysentiment)
summary(C5_Prediction)

# PCA Dataset Modeling w/ C5
set.seed(123)
C5_Fit_Galaxy_PCA <- C5.0(galaxysentiment~., trainGalaxyPCA,
                          rules = TRUE)
C5_Prediction_Galaxy_PCA <- predict(C5_Fit_Galaxy_PCA, testGalaxyPCA)
postResample(C5_Prediction_Galaxy_PCA, testGalaxyPCA$galaxysentiment)
summary(C5_Prediction_Galaxy_PCA)

# Original Dataset w/ recode()
GalaxyRC <- Galaxy
GalaxyRC$galaxysentiment <- as.factor(recode(GalaxyRC$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, 
                  '3' = 3, '4' = 4, '5' = 4))
str(GalaxyRC)
set.seed(123)
GalaxyRCtraining <- createDataPartition(GalaxyRC$galaxysentiment,
                                        p = 0.70, list = FALSE)
trainGalaxyRC <- GalaxyRC[GalaxyRCtraining,]
testGalaxyRC <- GalaxyRC[-GalaxyRCtraining,]
set.seed(123)
C5_Fit_Galaxy_RC <- C5.0(galaxysentiment~., data = trainGalaxyRC,
                         rules = TRUE)
C5_Prediction_Galaxy_RC <- predict(C5_Fit_Galaxy_RC, testGalaxyRC)
postResample(C5_Prediction_Galaxy_RC, testGalaxyRC$galaxysentiment)
summary(C5_Prediction_Galaxy_RC)
cmC5_GalaxyRC <- confusionMatrix(C5_Prediction_Galaxy_RC, testGalaxyRC$galaxysentiment)
cmC5_GalaxyRC

### Best Model: C5.0, Best Feature Selection Method: Original Data w/ Recode() ###
### Apply Model to Data ###
GalaxyLarge <- read.csv("GalaxyLargeMatrix.csv")
set.seed(123)
C5_Prediction_GalaxyLarge <- predict(C5_Fit_Galaxy_RC, GalaxyLarge)
str(C5_Prediction_GalaxyLarge)
write.csv(C5_Prediction_GalaxyLarge, "GalaxyLargePredicitons.csv")

###                           ###
### Report Plots & Statistics ###
###                           ###

iPhoneLargeMatrix <- read.csv("iPhoneLargeMatrix.csv")
iPhoneLargeMatrix$iphonesentiment <- as.factor(iPhoneLargeMatrix$iphonesentiment)
GalaxyLargeMatrix <- read.csv("GalaxyLargeMatrix.csv")
GalaxyLargeMatrix$galaxysentiment <- as.factor(GalaxyLargeMatrix$galaxysentiment)
summary(iPhoneLargeMatrix$iphonesentiment)
summary(GalaxyLargeMatrix$galaxysentiment)
iPhonePie <- data.frame(COM = c("Very Negative", 
                                       "Negative",
                                       "Positive",
                                       "Very Positive"),
                               values = c(12411, 710, 2103, 17228))
GalaxyPie <- data.frame(COM = c("Very Negative",
                                "Negative",
                                "Positive",
                                "Very Positive"),
                        values = c(12367, 708, 2074, 17303))
plot_ly(iPhonePie, labels = ~COM, values = ~values, type = "pie",
              textposition = 'inside',
              textinfo = 'label+percent',
              insidetextfont = list(color = '#FFFFFF'),
              hoverinfo = 'text',
              text = ~paste(values),
              marker = list(colors = colors,
                            line = list(color = '#FFFFFF', width = 1)),
              showlegend = F) %>% 
  layout(title = 'iPhone Sentiment',
         xaxis = list(showgrid = FALSE, zeroline = FALSE,
                      showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE,
                      showticklabels = FALSE))

plot_ly(GalaxyPie, labels = ~COM, values = ~values, type = "pie",
        textposition = 'inside',
        textinfo = 'label+percent',
        insidetextfont = list(color = '#FFFFFF'),
        hoverinfo = 'text',
        text = ~paste(values),
        marker = list(colors = colors,
                      line = list(color = '#FFFFFF', width = 1)),
        showlegend = F) %>% 
  layout(title = 'Galaxy Sentiment',
         xaxis = list(showgrid = FALSE, zeroline = FALSE,
                      showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, 
                      showticklabels = FALSE))
