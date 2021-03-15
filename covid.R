
####PLSC597 Method Tutorial: Regularization
####Presenter: Zicheng Cheng (zvc5199@psu.edu) 
####Codes are based on Rhys (2020), Chapter 11. 


##Deslatte (2020): To shop or shelter? Issue framing effects and social-distancing preferences in the COVID-19 pandemic. 
##Framing theory: Message frames and messenger type highly impact individual's health choice. 
##DV: Whether or not to delay shopping [no_shop]
##IV: health frame；CDC as messenger; President as messenger; State officials as messenger; Expert as messenger; CDC*H; Pres*H; State*H; Expert*H. 


##Install the packages, load the libraries and get the data. 
install.packages("lasso2")
install.packages("glmnet")
install.packages("randomForestSRC")
install.packages("kknn")
library(kknn)
library(tidyverse)
library(haven)
library(mlr)
library(lasso2)
library(glmnet)
library(randomForestSRC)
library(parallel)
library(parallelMap)

setwd("/Users/clairecheng/Desktop/PLSC597/regularization")
covid.data <- read_dta(file = "covid.dta")

###explore the dataset 
covidTib <- as_tibble(covid.data)
covidTib
summary(covidTib)

#plotting the relationship between each variable and dv. linear model fits 
covidUntidy <- gather(covidTib, "Variable", "Value", -no_shop)

ggplot(covidUntidy, aes(Value, no_shop)) +
  facet_wrap(~ Variable, scales = "free_x") +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw()

##Run the logistic model. 
mylogit <- glm(no_shop ~ gender + education + white + gop + shelter + jobloss + ideology_rs + health_frame + cdc_m + pres_m + state_m + expert_m + cdc_frame_h + pres_frame_h + state_frame_h + expert_frame_h, data = covidTib, family = "binomial")
summary(mylogit)

##Results show that gender, white, ideology, health_frame are significant predictors for choice not to shop. 
##When public-health issue frame instead of economic frame is used, people will be more likely to follow social-distancing guidance. 


######################1.Train the ridge regression model: 
##Creating Task and Learner 
covidTask <- makeRegrTask(data = covidTib, target = "no_shop")

ridge <- makeLearner("regr.glmnet", alpha = 0, id = "ridge")

##Generate and plot filter values 
filterVals <- generateFilterValuesData(covidTask)

plotFilterValues(filterVals) + theme_bw()

##gender, white, health frame and president frame have a positive contribution in terms of predicting DV. 
#But we are going to enter all those variables into the algorithm and let it shrink the ones that contribute less to the model.  
#Next we will tune the lambda parameters that control how big the penalty to apply to the parameter estimates. 
#Larger the lambda, the more parameters are shrunk toward zero. 

##Tuning the lambda's hyperparameter
ridgeParamSpace <- makeParamSet(
  makeNumericParam("s", lower = 0, upper = 15))

randSearch <- makeTuneControlRandom(maxit = 200)

cvForTuning <- makeResampleDesc("RepCV", folds = 3, reps = 10)


parallelStartSocket(cpus = detectCores())

tunedRidgePars <- tuneParams(ridge, task = covidTask,
                             resampling = cvForTuning,
                             par.set = ridgeParamSpace,
                             control = randSearch)

parallelStop()
tunedRidgePars

##Tune result:
##Op. pars: s=0.133  {the optimal lambda}
##mse.test.mean=0.185

##Plotting the hyperparameter tuning process. 
ridgeTuningData <- generateHyperParsEffectData(tunedRidgePars)

plotHyperParsEffect(ridgeTuningData, x = "s", y = "mse.test.mean",
                    plot.type = "line") +
  theme_bw()

##We need the lowest value. 



###########2. Train the Lasso model: 
#Create the task: 
covidTask2 <- makeRegrTask(data = covidTib, target = "no_shop")

#Set the learner: 
lasso <- makeLearner("regr.glmnet", alpha = 1, id = "lasso")

#Tuning lambda: 
lassoParamSpace <- makeParamSet(
  makeNumericParam("s", lower = 0, upper = 15))

parallelStartSocket(cpus = detectCores())

tunedLassoPars <- tuneParams(lasso, task = covidTask2,
                             resampling = cvForTuning,
                             par.set = lassoParamSpace,
                             control = randSearch)
parallelStop()

tunedLassoPars

##Tune result:
##Op. pars: s=0.038 {the optimal lambda}
##mse.test.mean=0.1889 

#plotting the tuning process: 
lassoTuningData <- generateHyperParsEffectData(tunedLassoPars)

plotHyperParsEffect(lassoTuningData, x = "s", y = "mse.test.mean",
                    plot.type = "line") +
  theme_bw()

##train the Lasso using the tuned lambda
tunedLasso <- setHyperPars(lasso, par.vals = tunedLassoPars$x)

tunedLassoModel <- train(tunedLasso, covidTask2)

##Extract the model parameters 
lassoModelData <- getLearnerModel(tunedLassoModel)

lassoCoefs <- coef(lassoModelData, s = tunedLassoPars$x$s)

lassoCoefs
###The result shows that, except for gender and health_frame, other parameter estimates are just dots. 
##Their slopes have been set to zero. Lasso have removed them from the model completely. This is how lasso can be used for feature selection. 


############3. Training elastic net 

elastic <- makeLearner("regr.glmnet", id = "elastic")

##tuning lambda and alpha. 
elasticParamSpace <- makeParamSet(
  makeNumericParam("s", lower = 0, upper = 10),
  makeNumericParam("alpha", lower = 0, upper = 1))

randSearchElastic <- makeTuneControlRandom(maxit = 400)

parallelStartSocket(cpus = detectCores())

tunedElasticPars <- tuneParams(elastic, task = covidTask,
                               resampling = cvForTuning,
                               par.set = elasticParamSpace,
                               control = randSearchElastic)

parallelStop()

tunedElasticPars

##Tune result:
##Op. pars: s=0.044; alpha=0.188 {optimal parameters}
##mse.test.mean=0.185

##plotting the tuning process 

elasticTuningData <- generateHyperParsEffectData(tunedElasticPars)

plotHyperParsEffect(elasticTuningData, x = "s", y = "alpha",
                    z = "mse.test.mean", interpolate = "regr.kknn",
                    plot.type = "heatmap") +
  scale_fill_gradientn(colours = terrain.colors(5)) +
  geom_point(x = tunedElasticPars$x$s, y = tunedElasticPars$x$alpha,
             col = "white") +
  theme_bw()

#Notice that the selected combination of lambda and alpha (the white dot) falls in a valley of mean MSE values, 
##suggesting our hyperparameter search space was wide enough.

############4. Benchmarking 
ridgeWrapper <- makeTuneWrapper(ridge, resampling = cvForTuning,
                                par.set = ridgeParamSpace,
                                control = randSearch)

lassoWrapper <- makeTuneWrapper(lasso, resampling = cvForTuning,
                                par.set = lassoParamSpace,
                                control = randSearch)

elasticWrapper <- makeTuneWrapper(elastic, resampling = cvForTuning,
                                  par.set = elasticParamSpace,
                                  control = randSearchElastic)

learners = list(ridgeWrapper, lassoWrapper, elasticWrapper, "regr.lm")

kFold3 <- makeResampleDesc("CV", iters = 3)

parallelStartSocket(cpus = detectCores())

bench <- benchmark(learners, covidTask, kFold3)

parallelStop()

bench

###Result: 
#task.id    learner.id mse.test.mean
#1 covidTib   ridge.tuned     0.1842876
#2 covidTib   lasso.tuned     0.1904702
#3 covidTib elastic.tuned     0.1855740
#4 covidTib       regr.lm     0.1834786

###Conclusion: OLS regression outperformed the regularization techniques. 

