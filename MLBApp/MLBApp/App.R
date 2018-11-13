#setwd("C:/Users/brigdani/Desktop/MLBApp")
source('init.R',local = TRUE)

# Server ------------------------------------------------------------------

server <- function(input, output,session) {
  
  #creates the initial state
  #initial state is a combination of random forests and knn applied to the tappy data set
  CVtune <- readRDS('initState.Rdata')
  makeReactiveBinding('CVtune')
  
  #take in the raw data
  #this will be influenced by which data set we choose
  rawdata <- reactive({
    
    if(is.null(input$file1)){
      datasets[[input$dataset]]
    } else {

    # input$file1 will be NULL initially. After the user selects
    # and uploads a file, head of that data file by default,
    # or all rows if selected, will be shown.
    
    req(input$file1)
    
    # when reading semicolon separated files,
    # having a comma separator causes `read.csv` to error
    tryCatch(
      {
        datasets[["newdata"]] <<- read.csv(input$file1$datapath,
                       header = TRUE,
                       sep = input$sep,
                       quote = input$quote)
      },
      error = function(e) {
        # return a safeError if a parsing error occurs
        stop(safeError(e))
      }
    )
    
    }
  })
  
  
  #we observe the data set we chose, specifically all the variables that pertain to it
  #with this we can select the dependent variable
  #what are we going to try to predict? 
  observe({
    updateSelectizeInput(session,'yvar',choices=names(rawdata()),selected = names(rawdata())[1])
  })
  
  #we observe the independent variables that pertain to the data set
  #We can select the independent variables we want to use in the data set
  #for some reason it crashes when I only have one variable
  observe({
    nms <- names(rawdata())[names(rawdata())!=input$yvar]
    data <- nms 
    names(data) <- nms
    updateSelectizeInput(session,'xvar',choices=data , selected = nms)
  })
  
  #initialize data sets
  #with ml we always split into test and training data set
#  dataTrain <- NULL
#  dataTest <- NULL
  
  #we make the data training set adjustable
  makeReactiveBinding('dataTrain')
  # makeReactiveBinding('dataTest')
  #our default task will be regression on a data set
  #this is arbitrary, easily could be a classification task
  modelType <- 'Regression'
  
  #This can change
  makeReactiveBinding('modelType')
  
  #where we allow the model type to be dynamic
  #sets the conditions for what will happen after we determine the modelType 
  #if it's a regression task, we will autoselect regression models
  #if its a classification task, we will autoselect classification models
  #interesting issue: 
  #should we have separate methods for binary and multiclass classification? 
  observeEvent(modelType,{
    #regression
    if(modelType=='Regression'){
      updateSelectizeInput(session,'slt_algo',choices = reg.mdls,selected = NULL)
    } 
    #classification
    else {
      if(modelType=='Classification'){
        updateSelectizeInput(session,'slt_algo',choices = cls.mdls,selected = NULL)
      } else{
        updateSelectizeInput(session,'slt_algo',choices = bin.mdls,selected = NULL)
      }
    }
  })
  
  
  #we have a yvariable 
  #we have our xvariables
  #we have our test split percentage
  observe({
    
    yvar <- input$yvar
    xvars <- input$xvar
    testsize <- input$sld_testsplit
    
    #breakdown
    if(is.null(yvar)||yvar==''){
      return(NULL)
    }
    
    # extract y and X from raw data
    # we first get y 
    y <- isolate(rawdata()[,yvar])
    
    # we next pull all the xvalues from the data 
    X <-  isolate(rawdata()[,xvars])
    
    # deal with NA values
    yi <- !is.na(y)
    Xi <- complete.cases(X)
    
    #merger of x and y where we have complete cases for both
    df2 <- cbind(y,X)[yi&Xi,]
    
    #determine the type of data that we are trying to predict
    c <- class(df2$y)
    
    #how many different values are in that variable 
    lvls <- length(unique(df2$y))
    
    #if we don't have many different data values
    #or if the data is a factor or character (strings)
    #then we do classification
    #otherwise we do regression
    if(lvls <= 6|(c!='numeric'& c!='integer')){
      
      if(lvls == 2){modelType <<- 'Binary'
      df2$y <- factor(df2$y)}
      
      else{modelType <<- 'Classification'
      df2$y <- factor(df2$y)}
      
    } else {
      modelType <<-'Regression'
      if(input$chk_logY){df2$y <- sign(df2$y) * log(abs(df2$y) + 0.001)}
    }
    
    #My favorite seed
    #This makes the data partitions consistent each and every time 
    set.seed(123456)
    sample_size <- nrow(df2) * (1-(testsize/100))
    trainIndex <- sample(1:nrow(df2), sample_size, replace = FALSE)
    #we isolate the test and training data sets
    #we do this by referencintg the training indices

    dataTrain <<- df2[ trainIndex,]
    dataTest  <<- df2[-trainIndex,]

  })
  
  
  
  #button functionality
  #when we hit the train button (register a click), we need to perform certain actions
  #we need to avoid hitting the button again (lock the button)
  #we need to select the parameters we want and run the models
  observeEvent(input$btn_train,{
    #lock the button
    #this releases when everything in here is done running
    disable('btn_train')
    on.exit(enable('btn_train'))
    
    #we select the models
    mdls <- isolate(input$slt_algo)
    
    #what type of cross validation are we doing
    #should I add a feature for repeated cross validation? 
    fitControl <- trainControl(method = "cv",
                               savePredictions = TRUE,
                               number = as.integer(input$rdo_CVtype),
                               allowParallel = FALSE,
                               search = 'random')
    
    #tuning mechanism
    #going to transition to a random search 
    #that is quicker, research supports it performs as well if not outperforming grid searches
    #Will have two types of features:
    #Coarse tuning: slider input range from 3 - 30
    #Fine tuning: singular input val of 100
    #some models obvi doesn't matter such as glm/lm, lda, qda
  
    tune <- isolate(input$slt_Tune)

      trainArgs <- list(
        
        'svmLinear'=list(form=y ~ .,
               data = dataTrain,
               preProcess = c('scale','center'),
               method = 'svmLinear',
               trControl = fitControl,
               tuneLength=tune),
        
        'svmPoly'= list(form=y ~ .,
                data = dataTrain,
                preProcess = c('scale','center'),
                method = 'svmPoly',
                trControl = fitControl,
                tuneLength=tune),
        
        'nnet'= list(form=y ~ .,
                  data = dataTrain,
                  preProcess = c('scale','center'),
                  method = 'nnet',
                  trControl = fitControl,
                  tuneGrid = expand.grid(decay=c(0.5, 0.1), size=c(4,5,6))),        
        
        'rf'=list(form=y ~ .,
                data = dataTrain,
                preProcess = c('scale','center'),
                method = 'rf',
                trControl = fitControl,
                tuneLength=tune,
                ntree=1e3),
        
        'knn'=list(form=y ~ .,
                 data = dataTrain,
                 preProcess = c('scale','center'),
                 method = 'knn',
                 trControl = fitControl,
                 tuneLength=tune),
        
        'nb'=list(form=y ~ .,
                data = dataTrain,
                preProcess = c('scale','center'),
                method = 'nb',
                trControl = fitControl,
                tuneLength=tune),
        
        'glm'=list(form=y ~ .,
                 data = dataTrain,
                 preProcess = c('scale','center'),
                 method = 'glm',
                 trControl = fitControl,
                 tuneLength=tune),
        
        'gam'=list(form=y ~ .,
                 data = dataTrain,
                 preProcess = c('scale','center'),
                 method = 'gam',
                 trControl = fitControl),

        'lda' = list(form=y ~ .,
                data = dataTrain,
                preProcess = c('scale','center'),
                method = 'lda',
                trControl = fitControl,
                tuneLength = 1),
        
        'qda' = list(form=y ~ .,
                data = dataTrain,
                preProcess = c('scale','center'),
                method = 'qda',
                trControl = fitControl),
        
        'svmRadial'= list(form=y ~ .,
                data = dataTrain,
                preProcess = c('scale','center'),
                method = 'svmRadial',
                trControl = fitControl,
                tuneLength=tune),
        'ada'=list(form=y ~ .,
                data = dataTrain,
                preProcess = c('scale','center'),
                method = 'ada',
                trControl = fitControl,
                tuneLength=tune),
        'adaboost'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'adaboost',
                   trControl = fitControl,
                   tuneLength=tune),
        'AdaBag'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'AdaBag',
                   trControl = fitControl,
                   tuneLength=tune),
        'bartMachine'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'bartMachine',
                   trControl = fitControl,
                   tuneGrid = expand.grid(num_trees = c(50), k = c(1,2), alpha = c(0.9, 0.95), beta = c(1), nu = c(2))),
        'bayesglm'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'bayesglm',
                   trControl = fitControl,
                   tuneLength=tune),
        'brnn'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'brnn',
                   trControl = fitControl,
                   tuneLength=tune),
        'bridge'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'bridge',
                   trControl = fitControl,
                   tuneLength=tune),
        'blassoAveraged'=list(form=y ~ .,
                      data = dataTrain,
                      preProcess = c('scale','center'),
                      method = 'blassoAveraged',
                      trControl = fitControl,
                      tuneLength=tune),
        'gamboost'=list(form=y ~ .,
                      data = dataTrain,
                      preProcess = c('scale','center'),
                      method = 'gamboost',
                      trControl = fitControl,
                      tuneLength=tune),
        'glmboost'=list(form=y ~ .,
                      data = dataTrain,
                      preProcess = c('scale','center'),
                      method = 'glmboost',
                      trControl = fitControl,
                      tuneLength=tune),
        'binda'=list(form=y ~ .,
                        data = dataTrain,
                        preProcess = c('scale','center'),
                        method = 'binda',
                        trControl = fitControl,
                        tuneGrid=data.frame(lambda.freqs = seq(0,1,0.1))),
        'LogitBoost'=list(form=y ~ .,
                        data = dataTrain,
                        preProcess = c('scale','center'),
                        method = 'LogitBoost',
                        trControl = fitControl,
                        tuneLength=tune),
        'C5.0'=list(form=y ~ .,
                        data = dataTrain,
                        preProcess = c('scale','center'),
                        method = 'C5.0',
                        trControl = fitControl,
                        tuneLength=tune),
        'rpart'=list(form=y ~ .,
                        data = dataTrain,
                        preProcess = c('scale','center'),
                        method = 'rpart',
                        trControl = fitControl,
                        tuneLength=tune),
        'glmboost'=list(form=y ~ .,
                        data = dataTrain,
                        preProcess = c('scale','center'),
                        method = 'glmboost',
                        trControl = fitControl,
                        tuneLength=tune),
        'C5.0Cost'=list(form=y ~ .,
                        data = dataTrain,
                        preProcess = c('scale','center'),
                        method = 'C5.0Cost',
                        trControl = fitControl,
                        tuneLength=tune),
        'rpartCost'=list(form=y ~ .,
                        data = dataTrain,
                        preProcess = c('scale','center'),
                        method = 'rpartCost',
                        trControl = fitControl,
                        tuneLength=tune),
        'cubist'=list(form=y ~ .,
                         data = dataTrain,
                         preProcess = c('scale','center'),
                         method = 'cubist',
                         trControl = fitControl,
                         tuneLength=tune),
        'enet'=list(form=y ~ .,
                         data = dataTrain,
                         preProcess = c('scale','center'),
                         method = 'enet',
                         trControl = fitControl,
                         tuneLength=tune),
        'xgbDART'=list(form=y ~ .,
                         data = dataTrain,
                         preProcess = c('scale','center'),
                         method = 'xgbDART',
                         trControl = fitControl,
                         tuneLength=tune),
        'xgbLinear'=list(form=y ~ .,
                         data = dataTrain,
                         preProcess = c('scale','center'),
                         method = 'xgbLinear',
                         trControl = fitControl,
                         tuneLength=tune),
        'xgbTree'=list(form=y ~ .,
                         data = dataTrain,
                         preProcess = c('scale','center'),
                         method = 'xgbTree',
                         trControl = fitControl,
                         tuneLength=tune),
        'fda'=list(form=y ~ .,
                         data = dataTrain,
                         preProcess = c('scale','center'),
                         method = 'fda',
                         trControl = fitControl,
                         tuneLength=tune),
        'gaussprLinear'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'gaussprLinear',
                   trControl = fitControl,
                   tuneLength=tune),
        'gaussprPoly'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'gaussprPoly',
                   trControl = fitControl,
                   tuneLength=tune),
        'gaussprRadial'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'gaussprRadial',
                   trControl = fitControl,
                   tuneLength=tune),
        'gamLoess'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'gamLoess',
                   trControl = fitControl,
                   tuneLength=tune),
        'bam'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'bam',
                   trControl = fitControl,
                   tuneLength=tune),
        'glmnet'=list(form=y ~ .,
                   data = dataTrain,
                   preProcess = c('scale','center'),
                   method = 'glmnet',
                  trControl = fitControl,
                  tuneLength=tune),
        'gbm_h2o'=list(form=y ~ .,
                      data = dataTrain,
                      preProcess = c('scale','center'),
                      method = 'gbm_h2o',
                      trControl = fitControl,
                      tuneLength=tune),
        'hdda'=list(form=y ~ .,
                      data = dataTrain,
                      preProcess = c('scale','center'),
                      method = 'hdda',
                      trControl = fitControl,
                      tuneLength=tune),
        'lssvmLinear'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'lssvmLinear',
                    trControl = fitControl,
                    tuneGrid=expand.grid(tau = c(1,2,3))),
        'lssvmPoly'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'lssvmPoly',
                    trControl = fitControl,
                    tuneGrid = expand.grid(tau = c(1,2,3), degree = c(1,2,3), scale = c(1,2,3))),
        'lda2'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'lda2',
                    trControl = fitControl,
                    tuneLength=tune),
        'manb'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'manb',
                    trControl = fitControl,
                    tuneLength=tune),
        'avNNet'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'avNNet',
                    trControl = fitControl,
                    tuneLength=tune),
        'mlpML'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'mlp',
                    trControl = fitControl,
                    tuneLength=tune),
        'mlpWeightDecayML'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'mlpWeightDecayML',
                    trControl = fitControl,
                    tuneLength=tune),
        'mlpSGD'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'mlpSGD',
                    trControl = fitControl,
                    tuneLength=tune),
        'awnb'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'awnb',
                    trControl = fitControl,
                    tuneLength=tune),
        'ORFlog'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'ORFlog',
                    trControl = fitControl,
                    tuneLength=tune),
        'ORFpls'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'ORFpls',
                    trControl = fitControl,
                    tuneLength=tune),
        'ORFridge'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'ORFridge',
                    trControl = fitControl,
                    tuneLength=tune),
        'ORFsvm'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'ORFsvm',
                    trControl = fitControl,
                    tuneLength=tune),
        'parRF'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'parRF',
                    trControl = fitControl,
                    tuneLength=tune),
        'pls'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'pls',
                    trControl = fitControl,
                    tuneLength=tune),
        'pda'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'pda',
                    trControl = fitControl,
                    tuneLength=tune),
        'pcr'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'pcr',
                    trControl = fitControl,
                    tuneLength=tune),
        'extraTrees'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'extraTrees',
                    trControl = fitControl,
                    tuneLength=tune),
        'rda'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'rda',
                    trControl = fitControl,
                    tuneLength=tune),
        'ridge'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'ridge',
                    trControl = fitControl,
                    tuneLength=tune),
        'spls'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'spls',
                    trControl = fitControl,
                    tuneLength=tune),
        'gbm'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'gbm',
                    trControl = fitControl,
                    tuneLength=tune),
        'svmBoundrangeString'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'svmBoundrangeString',
                    trControl = fitControl,
                    tuneLength=tune),
        'svmRadialWeights'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'svmRadialWeights',
                    trControl = fitControl,
                    tuneLength=tune),
        'svmExpoString'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'svmExpoString',
                    trControl = fitControl,
                    tuneLength=tune),
        'svmLinear2'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'svmLinear2',
                    trControl = fitControl,
                    tuneLength=tune),
        'svmRadialCost'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'svmRadialCost',
                    trControl = fitControl,
                    tuneLength=tune),
        'svmRadialSigma'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'svmRadialSigma',
                    trControl = fitControl,
                    tuneLength=tune),
        'svmSpectrumString'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'svmSpectrumString',
                    trControl = fitControl,
                    tuneLength=tune),
        'blasso'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'blasso',
                    trControl = fitControl,
                    tuneLength=tune),
        'lasso'=list(form=y ~ .,
                    data = dataTrain,
                    preProcess = c('scale','center'),
                    method = 'lasso',
                    trControl = fitControl,
                    tuneLength=tune),
        'qrf'=list(form=y ~ .,
                     data = dataTrain,
                     preProcess = c('scale','center'),
                     method = 'qrf',
                     trControl = fitControl,
                     tuneLength=tune),
        'qrnn'=list(form=y ~ .,
                      data = dataTrain,
                      preProcess = c('scale','center'),
                      method = 'qrnn',
                      trControl = fitControl,
                      tuneLength=tune),
        'rqlasso'=list(form=y ~ .,
                      data = dataTrain,
                      preProcess = c('scale','center'),
                      method = 'rqlasso',
                      trControl = fitControl,
                      tuneLength=tune),
        'rFerns'=list(form=y ~ .,
                      data = dataTrain,
                      preProcess = c('scale','center'),
                      method = 'rFerns',
                      trControl = fitControl,
                      tuneLength=tune),
        'cforest'=list(form=y ~ .,
                      data = dataTrain,
                      preProcess = c('scale','center'),
                      method = 'cforest',
                      trControl = fitControl,
                      tuneLength=tune),
        'lars'=list(form=y ~ .,
                       data = dataTrain,
                       preProcess = c('scale','center'),
                       method = 'lars',
                       trControl = fitControl,
                       tuneLength=tune)    
      ) 

      
    set.seed(123456)
      
    #automatically tunes all the models
    #will change such that can be ran in parallel
    #worked on code already
    #used n-1 cores to form cluster 
    #advantageous b/c we don't kill all other processes
    cl <- makeCluster(detectCores() - 1)
    registerDoParallel(cl)
      
    tune <- lapply(mdls,function(m){
      do.call('train',trainArgs[[m]])
    })
    
    #terminates the clusters
    #this is buggy
    #sometimes the clusters stay registered. We'd just need to reboot app then
    stopCluster(cl)
    
    
    #applies names to models that we ran
    #this allows for easy reference later
    names(tune) <- mdls
    
    #holds our information from running models
    CVtune <<- tune
    #saveRDS(CVtune,'initState.Rdata')
    
  })
  
  #process data
  #Takes the models that we ran and ultimately processes them so that we can rank them and compare them
  CVres <- reactive({
    
    #realistically doesn't ever happen but protect against
    if(is.null(CVtune)) return(NULL)
    
    #create a copy of CVtune so we can extract info w/out changing original
    fits <- CVtune
    #function takes fits
    #seperate logic exists to get different outputs based on model types
    getRes <- function(i){
      #get names of different models ie rf, glm, nb
      name <- names(fits)[i]
      
      #get result of models 
      res <- fits[[i]]$results
      
      #logic for regression
      if( isolate(modelType) == "Regression"){
        #extract results
        df <- res[(ncol(res)-5):ncol(res)]
        #take name of model, combine it with tuned hyper parameters to give an informative name
        #ie rf-3 would be a name for a random forest with mtry = 3
        #combine to form a single dataframe with model names and all results 
        apply(res,1,function(r) { 
          paste(c(round(as.numeric(r[1:(ncol(res)-6)]),digits = 7),sample(1:1000,1,replace = FALSE)),collapse = '-')

          }) %>% 
          paste(name,.,sep='-') -> model
        cbind.data.frame(model,df,name=name[[1]],stringsAsFactors =F)
      } 
      #EXACT SAME LOGIC FOR CLASSIFICATION EXCEPT SOME VALS CHANGED
      else {
        df <- res[(ncol(res)-3):ncol(res)]
        
        apply(res,1,function(r) paste(round(as.numeric(r[1:(ncol(res)-4)]),digits = 7),collapse = '-')) %>% 
          paste(name,.,sep='-') -> model
        cbind.data.frame(model,df,name=name[[1]],stringsAsFactors =F)
      }
    }
    
    #apply the function we just created to the data
    df <- plyr::ldply(1:length(fits),getRes)
    
    #how we will rank the models
    #ranking mechanism uses smallest to largest to rank
    #can use cumulative sum of ranks over all metrics to evaluate
    #RMSE, MAE on normal scale
    #RSquared, accuracy, kappa subtract from 1
    if(isolate(modelType)=='Regression'){
      df$rank <- rank(rank(df$RMSE)+rank(1-df$Rsquared) + rank(df$MAE),ties.method = 'first')
    } else {
      df$rank <- rank(rank(1-df$Accuracy)+rank(1-df$Kappa),ties.method = 'first')
    }
    #limit to 3 decimal places
    df[2:5] <- round(df[2:5],3)
    #order dataframe by results
    df[order(df$rank),]
  })
  
  #logic for getting the predicted values from our best models
  #takes the very best models that we have for each "FAMILY" of models
  #family broadly defined as randomforests, linear SVMS, etc 
  #as opposed to indivdual models: rf-2 svm-0.1 etc
  CVpredObs <- reactive({
    
    #what our the best fits we have? 
    fits <- CVtune
    
    #processes the best models to extract the predictions from the best model
    
    getObsPred <- function(i){
      # i <- 2
      #best model for each family specifically hyper parametrs
      bst <- fits[[i]]$bestTune
      
      #fits for that best model
      preds <- fits[[i]]$pred
      
      #name the predictions as the family of models
      preds$name <- names(fits)[i]
      
      #we are going to give a name to the model where we take our best model and we combine that with the model type
      preds$model <- paste(bst,collapse = '-') %>% paste(names(fits)[i],.,sep='-')
      
      
      ii <- lapply(1:length(bst),function(p){
        preds[names(bst)[p]]==as.character(bst[p][[1]])
      })
      
      if(length(bst)>1) data.frame(ii) %>% apply(.,1,all) -> ii else unlist(ii) ->ii
      preds[ii,-which(names(preds)%in%names(bst))]
    }
    
    df <- plyr::ldply(1:length(fits),getObsPred)
    str(df)
    # saveRDS(df,'CVpredObs.Rdata')
    df
    
  })
  
  #we group by the family of model
  #we then select the top model in each group
  topModels <- reactive({
    if(is.null(CVres()))
      return()
    CVres() %>% group_by(name) %>% filter(rank==min(rank)) -> df
    #selected the top models
    #we order them in a dataframe
    
    lst <- df$name[order(df$rank)]
    names(lst) <- df$model[order(df$rank)]
    lst
  }) 
  
  observe({
    #access the next top models
    lst <- topModels()
    #select which models we are looking at 
    updateSelectizeInput(session,'slt_Finalalgo',choices = lst,selected = lst[1])
    
  })
  
  testPreds <- reactive({
    #get the models
    #call them tune
    tune <- isolate(CVtune)
    if(is.null(tune)) return(NULL)
    #predict the values on the test data set
    #call the model and predict on test data set and cast to a dataframe
    lapply(CVtune[input$slt_Finalalgo],
           predict.train,isolate(dataTest)) %>% 
      data.frame() -> df
    
    # calculates important statistics about our best models
    # ie kapppa and accuracy for classification
    # ie Rsquared and MSE for our final model 
    if(isolate(modelType)=='Regression'){
      c <- apply(df[input$slt_Finalalgo],1,mean)
      
      s1 <- 1 - mean((dataTest$y-c)^2)/mean((dataTest$y-mean(dataTest$y))^2)
      s2 <- sqrt(mean((dataTest$y-c)^2))
      
    } else {
      c <- apply(df[input$slt_Finalalgo],1,modal)
      s1 <- sum(c==dataTest$y)/nrow(dataTest)
      s2 <- vcd::Kappa(table(c, dataTest$y))$Unweighted[1]
    }
    #compile into a list for exporting
    list(c=c,s1=s1,s2=s2)
    
  })
  
  
  
  
  # Outputs ---------------------------------------------------------------------
  

  output$varImp <- renderPlot({
    fit <- randomForest(y ~ . , data = dataTrain)
    var_importance <- data_frame(variable=setdiff(colnames(dataTrain), "y"),
                                 importance=as.vector(importance(fit)))
    
    var_importance <- arrange(var_importance, desc(importance))
    
    var_importance$variable <- factor(var_importance$variable, levels=var_importance$variable)
    pal <- wesanderson::wes_palette('FantasticFox1', type = 'c', n = length(var_importance$variable))[sample(1:length(var_importance$variable),length(var_importance$variable))]
    
    ggplot(var_importance, aes(x=variable, weight=importance, fill=variable)) +
      geom_bar()  +
      xlab("X Variable") + ylab("Variable Importance") + theme(axis.text.x=element_blank(),
                                                        axis.text.y=element_text(size=12),
                                                        axis.title=element_text(size=16),
                                                        plot.title=element_text(size=18),
                                                        legend.title=element_text(size=16),
                                                        legend.text=element_text(size=12)) + 
      scale_fill_manual(values = pal) + 
      guides(fill=guide_legend(title="Variable Name"))
    
  })
  
  #takes the results and lets us visualize the truth vs our predictions
  #plots test data with predicted values
  output$testsetPlot <- renderPlot({
    
    #make into a dataframe
    df <- data.frame(obs=dataTest$y,pred=testPreds()$c)
    
    #choose the color of my top model
    col <- pal[topModels()[[1]]]
    
    #if the model is regression
    #we will do a scatter plot of predicted values vs observed values
    #with a 1 to 1 line drawn which would represent the predicted and observed values matching exactly
    if(isolate(modelType)=='Regression'){
      
      #set some logical limits
      lims <- c(min(df$obs),max(df$obs))
      #make it a ggplot object
      ggplot(df) + 
        geom_abline(alpha=0.5) + #add the line
        geom_point(aes(x=obs,y=pred),color=col,size=2) + #include the points
        scale_x_continuous(limits = lims) +  #put it on a scale
        scale_y_continuous(limits = lims) +  #put it on a scale 
        coord_equal() + #make it a square
        theme_bw() +  #black and white theme (SIMPLE)
        xlab('Observed') +  #Observed
        ylab('Predicted') +  #Predicted
        theme(legend.position='none') #No Legend
    } else {
      # for classification
      #get the observed values 
      df$pred <- factor(df$pred,levels=levels(df$obs))
      
      #pipe dataframe into tidyverse summarize and get counts of predicted and observed for each group
      df %>% group_by(pred,obs) %>% 
        summarise(n=n()) %>% #this creates the confusion matrix
        ggplot(.) + #piped into ggplot 
        geom_raster(aes(x=obs,y=pred,alpha=n),fill=col) + #here we use a raster plot (HEAT MAP to make it pretty)
        geom_text(aes(x = obs,y = pred,label = n)) + 
        coord_equal() + 
        theme_bw() +
        xlab('Observed') +
        ylab('Predicted') +
        theme(legend.position = 'none')
      
    }
    
  })
  
  #test set predictions and evaluation
  output$testsetS1 <- renderValueBox({
    
    #so we can do either regression or classification
    lab <- ifelse(isolate(modelType)=='Regression','Variance explained','Accuracy')
    
    #takes the test predictions such as accuracy or Rsquared and gives us info on that
    valueBox(paste(round(testPreds()$s1*100,1),'%'),lab,icon = icon('cube'))
    
  })
  
  output$testsetS2<- renderValueBox({
    
    #so we can do either regression or classification
    lab <- ifelse(isolate(modelType)=='Regression','RMSE','Kappa')
    
    #takes the test predictions such as RMSE or Kappa and gives us info on that
    valueBox(round(testPreds()$s2,3),subtitle = lab,icon = icon('cube'))
  
  })
  
  
  
  #gives me a beautiful raw data format where we can directly look at our data
  output$rawdata <- renderDataTable({rawdata()},
                                    options = list(pageLength = 10,searching = FALSE))
  
  #some pretty formating where we can look at our models 
  #This is under the CV stats tab
  output$model_info <- renderDataTable({
    CVres()[c(1:7)]
    
  },    options = list(rowCallback = I(
    lapply(1:length(mdls),function(i) tableCSS(mdls[i],pal[i])) %>% 
      unlist %>% 
      paste(.,collapse = '') %>% 
      paste('function(row, data) {',.,'}')
  ),
  pageLength = 10,searching = FALSE
  )
  )
  
  #gets the best model
  #tells if its a classification task
  #depending on task, plots data differently
  output$CVplot2 <- renderPlot({
    
    #get the model type
    type <- isolate(modelType)
    
    #gets the data on the best models
    df <-CVpredObs()
    
    #if we are doing regression we will plot predicted vs observed
    if(type=='Regression'){
      
      #set some logical limits
      lims <- c(min(df$obs),max(df$obs))
      
      #make a ggplot object
      #this task is identical to the task previously
      ggplot(df) +
        geom_abline(alpha=0.5) + 
        geom_point(aes(x=obs,y=pred,col=name)) +
        scale_x_continuous(limits = lims) +
        scale_y_continuous(limits = lims) +
        scale_color_manual(values=pal) +
        coord_equal() +
        facet_wrap(~name) +
        theme_bw() +
        xlab('Observed') +
        ylab('Predicted') +
        theme(legend.position='none')
    } else {
      df %>% group_by(pred,obs,name) %>% 
        summarise(n=n()) %>% #make a confusion matrix
        ggplot(.) +
        geom_raster(aes(x=obs,y=pred,fill=name,alpha=n)) +
        geom_text(aes(x=obs,y=pred,label=n)) +
        scale_fill_manual(values=pal) +
        coord_equal() +
        facet_wrap(~name) +
        theme_bw() +
        xlab('Observed') +
        ylab('Predicted') +
        theme(legend.position='none')
      
    }
  })
  
  #plots the models with error bars on accuracy
  output$CVplot1 <- renderPlot({
    resdf <- CVres()
    type <- isolate(modelType)
    
    sltN <- min(nrow(resdf),input$slt_nMod) 
    resdf <- resdf[1:sltN,]  
    resdf$model <- factor(resdf$model,levels = rev(resdf$model[resdf$rank]))
    if(type=='Regression'){
      ggplot(resdf,aes(x=model,color=name)) +
        geom_errorbar(aes(ymin=RMSE-RMSESD,ymax=RMSE+RMSESD),size=1) +
        geom_point(aes(y=RMSE),size=3) +
        scale_color_manual(values=pal) +
        coord_flip() +
        theme_bw() +
        xlab('') +
        theme(legend.position='none') -> p1
      ggplot(resdf,aes(x=model,color=name)) +
        geom_errorbar(aes(ymin=Rsquared-RsquaredSD,ymax=Rsquared+RsquaredSD),size=1) +
        geom_point(aes(y=Rsquared),size=3) +
        scale_color_manual(values=pal) +
        coord_flip() +
        theme_bw() +
        xlab('') +
        theme(legend.position='none') -> p2
     
       ggplot(resdf,aes(x=model,color=name)) +
        geom_errorbar(aes(ymin=MAE-MAESD,ymax=MAE+MAESD),size=1) +
        geom_point(aes(y=MAE),size=3) +
        scale_color_manual(values=pal) +
        coord_flip() +
        theme_bw() +
        xlab('') +
        theme(legend.position='none') -> p3 
      
      #plot the ggplot objects
      #and MAE for regression
      gridExtra::grid.arrange(p2,p1,p3,ncol=3)
      
      
    } else {
      ggplot(resdf,aes(x = model,color = name)) +
        geom_errorbar(aes(ymin = Kappa - KappaSD, ymax = Kappa + KappaSD),size=1) +
        geom_point(aes(y = Kappa),size = 3) +
        scale_color_manual(values = pal) +
        coord_flip() +
        theme_bw() +
        xlab('') +
        theme(legend.position='none') -> p1
      ggplot(resdf,aes(x= model,color = name)) +
        geom_errorbar(aes(ymin= Accuracy- AccuracySD, ymax = Accuracy + AccuracySD),size = 1) +
        geom_point(aes(y=Accuracy),size = 3) +
        scale_color_manual(values = pal) +
        coord_flip() +
        theme_bw() +
        xlab('') +
        theme(legend.position = 'none') -> p2
      
      #plot the ggplot objects
      #want to add them for F1/Precision/Recall for classification
      gridExtra::grid.arrange(p2,p1,ncol=2)
    }
    


    
  })
  
  output$downloadData <- downloadHandler(
    filename = function(){
      paste('data-', Sys.Date(), '.csv', sep='')
    },
    content = function(file) {
      data <- CVres()
      write.csv(data, file)
    }
  )
  
  #Print the data type in the select data step
  output$Ytype <- renderText(class(dataTrain$y))
  
  #print the name of the data set
  output$txt_dataset <- renderPrint(cat('Dataset:',input$dataset))
  
  #print the number of observations in the raw data
  output$txt_n <- renderPrint(cat('n obs:',nrow(rawdata())))
  
  #print the yvar what are we trying to predict
  output$txt_Yvar <- renderPrint(cat('Y var:',input$yvar))
  
  #print the % split of the test set
  output$txt_testSet <- renderPrint(cat('Test set:',input$sld_testsplit,'%'))
  
  #print the task
  output$txt_Type <- renderPrint(cat('Model Type:',modelType))
  
  #print the number of folds
  output$txt_CV <- renderPrint(cat('CV folds:',input$rdo_CVtype))
  
  #print the number of models we trained
  output$txt_nModels <- renderPrint(cat('Models trained:',nrow(CVres())))
  
  #Which is the best model? 
  output$txt_bestModel <- renderPrint(cat('Best Model:',(CVres()$model[1])))
  
  #print the statistics from the best model
  output$txt_bestModelStat1 <- renderPrint({
    #if the model is regression we print RSquared
    if(modelType=='Regression'){
      cat('Variance Explained:',(CVres()$Rsquared[1]*100),'%')
    } 
    #if the model is classification, we print accuracy
    else {
      cat('Accuracy:',(CVres()$Accuracy[1]))
    }
  })
  
  #print the statistics from the best model
  output$txt_bestModelStat2 <- renderPrint({
    #if the model is regression, we print RMSE
    if(modelType=='Regression'){
      cat('RMSE:',(CVres()$RMSE[1]))
    } 
    #if the model is classification we print KAPPA 
    else {
      cat('Kappa:',(CVres()$Kappa[1]))
    }
  })
  
  
  
  #prints the summary of the y data in the training set
  #this is controlled with a seed 
  output$Ystats <- renderPrint({
    
    summary(dataTrain$y)
    
  })
  

  #prints the distribution of the y variable
  output$Yplot <- renderPlot({
    
    #if the model is regression
    if(modelType=='Regression'){
      
      #kernelized density with a basic fill
      #we can roughly see what the distribution is like
      ggplot(dataTrain,aes(x=y))+
        geom_density(alpha=0.7,adjust=0.5,fill="#5BBCD6")+
        theme_bw()+
        ggtitle('Y Distribution')+
        xlab('')
      
      
      # wes_palettes$Darjeeling
      
      
    } 
    
    #if the model is classification
    #we print a bar plot which shows us how many objects we have in each class and how many classes we have
    else {
      pal <- wes_palette('Darjeeling1',n = length(unique(dataTrain$y)),type = 'c')
      ggplot(dataTrain,aes(x=y,fill=y))+
        geom_bar(stat='count')+
        scale_fill_manual(values=pal)+
        xlab('')+
        ggtitle('Y Class Frequency')+
        coord_flip()+
        theme(legend.position='none')
    }
    
  })
  
}





# UI ----------------------------------------------------------------------

ui <- bootstrapPage(useShinyjs(),
                    # Add custom CSS & Javascript
                    tagList(tags$head(
                      tags$link(rel="stylesheet", type="text/css",href="style.css"),
                      tags$script(type="text/javascript", src = "busy.js"),
                      lapply(1:length(mdls),function(i) modelCSS(mdls[i],pal[i]))
                      
                    )),
                    
                    
                    tagList(tags$head(tags$style(
                        "
                        txt_bestModel, txt_bestModelStat1, txt_bestModelStat2,
                        txt_Type, txt_CV, txt_nModels{
                        font-size: 75%;
                        }"
                      ))),
                    
                    dashboardPage(skin = 'green',
                      dashboardHeader(title = HTML(paste(icon('cogs'),'ML Benchmarking'))
                      ),
                      dashboardSidebar(
                        sidebarMenu(
                          # Setting id makes input$tabs give the tabName of currently-selected tab
                          id = "tabs",
                          menuItem("Step 1: Select Data", tabName = "setup", icon = icon("folder"),selected = TRUE),
                          menuItem("Step 2: Select & Train Models", tabName = "model", icon = icon("cog")),
                          menuItem("Step 3: Evaluate Model", tabName = "test", icon = icon("signal"))
                        ),
                        hr(),
                        fluidRow(
                          column(width=1),
                          column(width=10,
                                 h5(textOutput('txt_dataset')),
                                 h5(textOutput('txt_n')),
                                 h5(textOutput('txt_Yvar')),
                                 h5(textOutput('txt_testSet'))
                                 
                                 
                          ),
                          column(width=1)
                        ),
                        downloadButton('downloadData', label = "Save task report")
                        
                                         
                      ),
                      dashboardBody(
                        tabItems(
                          tabItem("setup",
                                  box(width = 4,title = 'Data',solidHeader = TRUE,status = 'primary',
                                      selectInput('dataset',label = 'Choose Data Source',
                                                  choices = names(datasets),selected='iris'),
                                      actionButton('btn_viewData',label = 'View Data',icon=icon('table')),
                                      hr(),
                                      fileInput("file1", "Choose CSV File",
                                               multiple = FALSE,
                                               accept = c("text/csv",
                                                          "text/comma-separated-values,text/plain",
                                                          ".csv")),
                                      
                                      # Horizontal line ----
                                      tags$hr(),

                                      # Input: Select separator ----
                                      radioButtons("sep", "Separator",
                                                   choices = c(Comma = ",",
                                                               Semicolon = ";",
                                                               Tab = "\t"),
                                                   selected = ","),
                                      
                                      # Input: Select quotes ----
                                      radioButtons("quote", "Quote",
                                                   choices = c(None = "",
                                                               "Double Quote" = '"',
                                                               "Single Quote" = "'"),
                                                   selected = '"'),
                                      sliderInput('sld_testsplit',label = label.help('Test set %','lbl_testsplit'),
                                                  min = 15,max = 80,step = 1,value = 33),
                                      bsTooltip(id = "lbl_testsplit", title = "% of data to set aside for test data", 
                                                placement = "right", trigger = "hover")
                                      
                                      
                                  ),
                                  box(width=4,title = 'y variable',solidHeader = TRUE,status = 'primary',
                                      helpText('Select the variable we would like to predict'),
                                      selectizeInput('yvar',label=label.help('y var','lbl_yvar'),choices = character(0)),
                                      helpText(HTML(paste('data type:', textOutput('Ytype')))),
                                      bsTooltip(id = "lbl_yvar", title = "Variable to predict", 
                                                placement = "right", trigger = "hover"),
                                      hr(),
                                      plotOutput('Yplot',height=260),
                                      conditionalPanel("output.Ytype == 'numeric'|output.Ytype == 'integer'",
                                                       checkboxInput('chk_logY',label = 'log transform')
                                      ),
                                      verbatimTextOutput('Ystats')
                                      
                                  ),
                                  box(width=4,title = 'X vars',solidHeader = TRUE,status = 'primary',
                                      selectizeInput('xvar',label=label.help('X (Predict Y as function of):','lbl_xvar'),choices = character(0),multiple = TRUE),
                                      bsTooltip(id = "lbl_xvar", title = "Try and predict Y as function of these variables", 
                                                placement = "right", trigger = "hover"), 
                                      hr(),
                                      h4("Variable Importance"),
                                      plotOutput('varImp',height = 360)
                                      ### Will insert a section on variable importance here
                                      ### Can use this to help inform people of which variables to include
                                      ### Will inclue a blurb about it in help file
                                      ### Devils in the details
                                      
                                  ),
                                  bsModal('data',title = 'Dataset',trigger = 'btn_viewData',size = 'large',
                                          dataTableOutput('rawdata')
                                  )
                          ),
                          tabItem("model",

                                  column(width=3,
                                         box(width = 12,title = 'Model Options',solidHeader = TRUE,status = 'primary',
                                             selectInput('slt_algo',label = 'Algorithm:'%>%label.help('lbl_algo'),
                                                         choices = reg.mdls,selected = reg.mdls,multiple=T),
                                             sliderInput('slt_Tune','Parameter Tuning'%>%label.help('lbl_Tune'),
                                                            min = 3, max = 50, step = 1, value = 10, width='75%'),
                                             sliderInput('slt_nMod','Number of Models to Show'%>%label.help('lbl_nMod'),
                                                         min = 1, max = 25, step = 1, value = 5, width='75%'),
                                             radioButtons('rdo_CVtype',label = 'Cross-validation folds'%>%label.help('lbl_CV'),
                                                          choices = c('3-fold'=3,'5-fold'=5,'10-fold'=10),inline = TRUE),
                                             
                                             actionButton('btn_train',label = 'Train Models',
                                                          icon = icon('bullseye'),#'bullseye','rocket'
                                                          class='btn-danger fa-lg',
                                                          width='100%'),
                                             bsTooltip(id = "lbl_algo", title = "Which algorithms to test", 
                                                       placement = "right", trigger = "hover"),
                                             bsTooltip(id = "lbl_Tune", title = "Type of tuning which is performed to optimize model parameters", 
                                                       placement = "right", trigger = "hover"),
                                             bsTooltip(id = "lbl_nMod", title = "How many models to display in the graphical output comparing metrics", 
                                                                                                         placement = "right", trigger = "hover"),
                                             bsTooltip(id = "lbl_CV", title = "Number of splits of training data used to tune parameters", 
                                                       placement = "right", trigger = "hover")
                                             
                                         ),
                                         box(width = 12,title = 'Summary',solidHeader = FALSE,
                                             status = 'primary',
                                             helpText(textOutput('txt_bestModel'), style = 'font-size:13px'),
                                             helpText(textOutput('txt_bestModelStat1'), style = 'font-size:13px'),
                                             helpText(textOutput('txt_bestModelStat2'), style = 'font-size:13px'),
                                             helpText(textOutput('txt_Type'), style = 'font-size:13px'),
                                             helpText(textOutput('txt_CV'), style = 'font-size:13px'),
                                             helpText(textOutput('txt_nModels'), style = 'font-size:13px')
                                         )
                                  )
                                  ,
                                  tabBox(width = 9,
                                         tabPanel(title = 'CV Model Rank',#icon = icon('sort-amount-asc'),
                                                  h4('Cross-validation results'),
                                                  plotOutput('CVplot1',height=600)
                                         ),
                                         tabPanel(title = 'CV Pred vs Obs',
                                                  h4('Observed vs Predicted (best candidate for algorithm)'),
                                                  plotOutput('CVplot2',height=600)
                                         ),
                                         tabPanel(title = 'CV Stats',
                                                  h4('Performance statistics from cross-validation'),
                                                  
                                                  dataTableOutput('model_info')
                                         )
                                  )
                          ),
                          tabItem("test",
                                  column(width=3,
                                         box(width = 12,title = 'Test Set Predictions',solidHeader = FALSE,status = 'primary',                                              selectInput('slt_Finalalgo',label = 'Final Model:'%>%label.help('lbl_Finalalgo'),
                                                         choices=mdls,multiple=T),
                                             helpText('The best cross-validated model is selected by default. 
                                                      Multiple models can be selected to make ensemble predictions'),
                                             bsTooltip(id = "lbl_Finalalgo", title = "Which algorithms to use to predict test", 
                                                       placement = "right", trigger = "hover")
                                             
                                             ),
                                         valueBoxOutput('testsetS1',width=12),
                                         valueBoxOutput('testsetS2',width=12)
                          ),
                          box(width = 6,title = 'Test Set observed vs Predicted',
                              solidHeader = TRUE,status = 'primary',
                              plotOutput('testsetPlot')
                          )
                                  )
                        )
                        )
                      
                      
                    )
)


#runApp(appDir = list(ui = ui, server = server))
shinyApp(ui = ui, server = server)