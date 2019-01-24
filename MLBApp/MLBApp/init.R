options(repos = c(CRAN = "https://cran.rstudio.com"))
#setwd("C:/Users/brigdani/Desktop/MLBApp")

library(shiny)
require(shinyBS)
require(shinydashboard)
require(shinyjs)
require(caret)
require(plyr)
require(dplyr)
require(tidyr)
require(Cairo)
require(raster)
require(gstat)
require(wesanderson)
require(nnet)
require(randomForest)
require(gridExtra)
require(doParallel)
library(parallel)
# We will inevitably add many more packages. 
# Packages that will be added will depend on models ultimately selected
# It is impractical to use all the models, so we will only select the most well known/most UTD models
# car, foreach, methods, plyr, nlme, reshape2, stats, stats4, utils, grDevices


# Not all of these are required but shinyapps.io was crashing and 
# importing one of these solved the issue
require(kernlab)
require(klaR)
require(vcd)
require(e1071)
require(gam) #
require(ipred) 
require(MASS)
require(ellipse) #
require(mda) #
require(mgcv) 
require(mlbench) #
require(party) #
require(MLmetrics) #
require(Cubist) #
require(testthat)
require(ada) #
require(fastAdaboost) #
require(adabag) #
require(bartMachine) #
require(arm) 
require(brnn) #
require(monomvn) #
require(mboost) #
require(import) #
require(caTools)
require(earth) #
require(mda)  
require(C50) #
require(HDclassif) #
require(RSNNS) #
require(caret)
require(bnclassify) #
require(FCNN4R) #
#require(obliqueRF) 
require(pls) #
require(mda)
#require(rFerns)
require(extraTrees) #
require(spls) #
#require(gbm)
require(elasticnet) #
#require(quantregForest)
#require(rqPen)
#require(qrnn)
library(digest)

train <- caret::train
importance <- randomForest::importance

Tappy <- read.delim("TappyDemo.tsv")
Microbes <- read.delim("Microbes.tsv")
HealthSurvey <- read.delim("MHSurv.tsv")
SNPLung <- read.delim("SNPLungCancer.tsv")
Pharynx <- read.csv("Pharynx.csv",stringsAsFactors = TRUE)
BreastTumor <- read.csv("BreastTumor.csv",stringsAsFactors = TRUE)
LowBWT <- read.csv("LowBwt.csv",stringsAsFactors = TRUE)
Nursery <- read.csv("Nursery.csv",stringsAsFactors = TRUE)
Readmission <- read.csv("Readmission.csv",stringsAsFactors = TRUE)
EchoMonths <- read.csv("EchoMonths.csv",stringsAsFactors = TRUE)


# Data sets that will be considered
# This will be a comprohensive list of a variety of data sets
# Different tasks can be applied to each data set
# In all, I imagine this to be a set of perhaps 20 data sets
# These data sets will represent a variety of topics based on real world, simulated data, and toy data
# These data sets will be made of claims, EMR, questionnaire/public use, wearable/patient data, and genetics/genomics data
# This section will be easily expandable

datasets <- list(
  'iris' = iris,
  'Tappy' = Tappy,
  'Microbes'= Microbes,
  'Health Survey' = HealthSurvey,
  'metaSNP' = SNPLung,
  'Pharynx' = Pharynx,
  'Breast Tumor' = BreastTumor,
  'Low BWT' = LowBWT,
  'Nursery' = Nursery,
  'Readmission' = Readmission,
  'Echo Months' = EchoMonths

)

# Tuneable parameters
# These will be sensitive to model choice
# Parameters will be stored in a named list
# When that model is called, so will the tuneable parameters of that model
# The tuneable parameters will be simple in the short term; long term this will be an exhaustive process
# Tuneable parameters will be based on tuneable parameters of selected models supported by Caret
# Some parameters will be restricted such as but not explicitly Gaussian kernels in kernel based methods
# An illustrative but simple example: 
# KNN is a simple algorithm based on predictions of the K nearest points based on Euclidean distance
# A simple grid search will be performed to locate the optimal number of neighbors to consider
# The final value will be used for prediction

# These are the codes based on Caret for the call to each model from each package
# In some cases, there may be multiple models of the same type that have multiple calls
mdls <- list('svmLinear'='svmLinear',#1 WORKS
             'svmPoly'='svmPoly', #2 WORKS
             'Neural Network'='nnet', #3 WORKS
             'randomForest'='rf', #4 WORKS
             'k-NN'='knn', #5 WORKS
             'Naive Bayes'='nb', #6 WORKS
             'GLM'='glm', #7 WORKS
             'GAM'='gam', #8 WORKS
             'Boosted Classification Trees' = 'ada', #9 WORKS
             'LDA' = 'lda', #10 WORKS 
             'QDA' = 'qda', #11 WORKS
             'svmRadial' = 'svmRadial',#12
             'AdaBoost Classification Trees' = 'adaboost', #13 add WORKS
             'Bagged AdaBoost' = 'AdaBag', #14 add WORKS
             'Bayesian Additive Regression Trees' = 'bartMachine', #15 add DOESN'T Work
             'Bayesian GLM' = 'bayesglm', #16 add WORKS
             'Bayesian Regularized Neural Network' = 'brnn', #17 add WORKS
             'Bayesian Ridge Regression' = 'bridge', #18 add WORKS
             'Bayesian Ridge Regression - Averaged' = 'blassoAveraged', #19 add WORKS
             'Binary Discriminant Analysis' = 'binda', #20 add DOES NOT WORK
             'Boosted Generalized Additive Model' = 'gamboost', #21 add DOES NOT WORK
             'Boosted GLM' = 'glmboost', #22 add WORKS
             'Boosted Logistic Regression' = 'LogitBoost', #23 add WORKS
             'C5.0' = 'C5.0', #24 add WORKS
             'CART' = 'rpart', #25 add WORKS
             'Cost-Sensitive C5.0' = 'C5.0Cost', #26 add WORKS
             'Cost-Sensitive CART' = 'rpartCost', #27 add WORKS
             'Cubist' = 'cubist', #28 add WORKS
             'Elasticnet' = 'enet', #29 add WORKS
             'xgbDART' = 'xgbDART', #30 add WORKS
             'xgbLinear' = 'xgbLinear', #31 add WORKS
             'xgbTree' = 'xgbTree', #32 add WORKS
             'Gaussian Process' = 'gaussprLinear', #33 add WORKS
             'Gaussian Process-Polynomial Kernel' = 'gaussprPoly', #34 add WORKS
             'Gaussian Process-Radial Basis Kernel' = 'gaussprRadial', #35 add WORKS
             'GAM-Spline' = 'bam', #36 add WORKS
             'GAM-Loess' = 'gamLoess', #37 add WORKS
             'glmnet' = 'glmnet', #38 add WORKS
             'Conditional Inference randomForest' = 'cforest', #39 add WORKS
             'FDA' = 'fda', #40 add WORKS,
             'Gradient Boosting Machines' = 'gbm_h2o', #41 DOESN'T WORK
             'High Dimensional Discriminant Analysis' = 'hdda', #42 WORKS
             'Least Squares SVM' = 'lssvmLinear', #43 DOESN'T WORK
             'Least Squares SVM-Poly' = 'lssvmPoly', #44 DOESN'T WORK
             'LDA-2' = 'lda2', #45 WORKS
             'Model Averaged Naive Bayes' = 'manb', #46 DOESNT WORK
             'Model Averaged Neural Network' = 'avNNet', #47 WORKS
             'Multi-Layer Perceptron' = 'mlpML', #48 WORKS
             'Multi-Layer Perceptron-Weight Decay' = 'mlpSGD', #49 WORKS
             'Oblique randomForest-log' = 'ORFlog', #50
             'Oblique randomForest-pls' = 'ORFpls', #51 
             'Oblique randomForest-ridge' = 'ORFridge', #52
             'Oblique randomForest-svm' = 'ORFsvm', #53
             'Parallel randomForest' = 'parRF', #54
             'Partial Least Squares' = 'pls', #55 
             'Penalized Discriminant Analysis' = 'pda', #56 
             'Principal Component Analysis' = 'pcr', #57
             'Extra Trees' = 'extraTrees', #58
             'Regularized Discriminant Analysis' = 'rda', #59
             'Ridge Regression' = 'ridge', #60
             'Sparse Partial Lease Squares' = 'spls', #61
             'Stochastic Gradient Boosting' = 'gbm', #62
             'SVM-Boundrange String Kernel' = 'svmBoundrangeString', #63
             'SVM-Radial Weights' = 'svmRadialWeights', #64
             'SVM-Linear2' = 'svmLinear2', #65
             'SVM-Expo String' = 'svmExpoString', #66
             'SVM-Radial Cost' = 'svmRadialCost', #67 
             'SVM-Radial Sigma' = 'svmRadialSigma', #68
             'SVM-Spectrum String' = 'svmSpectrumString', #69
             'Bayesian Lasso' = 'blasso', #70
             'Lasso' = 'lasso', #71
             'Quantile randomForest' = 'qrf', #72
             'Quantile Regression Neural Network' = 'qrnn', #73
             'Penalized Quantile Regression' = 'rqlasso', #74
             'Random Ferns' = 'rFerns', #75
             'Least Angles Regression' = 'lars'
)
# This is a list of whether these are classification or regression models
# Some models will both be classification and regression models
# some models had to be removed
# for some reason, not all metrics are supported in Caret 

mdli <- list( # insert A TRUE after the second T for NNET
  'Regression'=c(T,T,F,T,T,F,T,F,F, FALSE ,F,T,F,F,F,FALSE,FALSE,T,TRUE,F,F,T,F,F,T,F,F,T,T,F,F,F,F,F,F,F,F,F,T,FALSE,F,F,F,F,F,F,F, T,T #49 at the end of this line
                 ,F,F,F,F,F,T,F,T,T,F,T,T,F,F,F,T,F,T,T,F,T,T,F,F,F,F,T), #TRUE is at 19, FALSE is at 40 
  'Classification'=c(T,T,T,T,T,T,F,F,F, FALSE ,T,T,F,T, F,FALSE ,FALSE,F,FALSE,F,F,F,F,T,T,F,F,F,F,F,F,F,F,F,F,F,F,F,TRUE,F, F,F,F,F,F,F,T, T,T   #49 at the end of this line
                     ,F,F,F,F,F,F,T,F,T,FALSE,F,F,F,F,F,T,F,T,T,F,F,F,F,F,F,F,F), #FALSE is at 19, TRUE is at 39
  'Binary'= c(T,T,T,T,T,T,T,F,T, FALSE ,T,T,T,T,F,FALSE ,FALSE,F,FALSE,F, FALSE ,F,T,T,T,T,T,F,F,F,F,F,F,F,F,F,F,F,TRUE,F,F,F,F,F,F,F,T, T,T #49 at the end of this line
              ,F,F,F,F,F,F,T,F,T,FALSE,F,F,F,F,T,T,F,T,T,F,F,F,F,F,F,F,F) #FALSE is at 19, TRUE is at 39
)  

# This subsets the models to be classification or regression 
(reg.mdls <- mdls[mdli[['Regression']]])
(cls.mdls <- mdls[mdli[['Classification']]])
(bin.mdls <- mdls[mdli[['Binary']]])


# We will take a random sample of colors 
# We will assign the models we are working with to different colors
# This section will need to be expanded when we have more models
# At that point we can transition to RColorBrewer and use a known template
# We should be aware that at some point once many models are enabled, we will have trouble with a comprehensive color scheme

pal <- wesanderson::wes_palette("FantasticFox1",length(mdls), type = 'continuous')
pal <- pal[sample(1:length(pal),length(pal))]
names(pal) <- mdls

# Assigns the color to the model selection
# Here we color code the models 
modelCSS <-   function(item,col){
  tags$style(HTML(paste0(".selectize-input [data-value=\"",item,"\"] {background: ",col," !important}")))
}

# Assigns the color to the table
# Here we color code the model information on the table
tableCSS <- function(model,col){
  paste0('if (data[6] == "',model,'")
         $("td", row).css("background", "',col,'");')
}  

# Creates the (?) button
# This will provide information about each object
label.help <- function(label,id){
  HTML(paste0(label,actionLink(id,label=NULL,icon=icon('question-circle'))))
}
