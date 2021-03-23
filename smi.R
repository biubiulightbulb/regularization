#PLSC597 Method Tutorial- Regularization 03/23
#Presenter: Zicheng Cheng [zvc5199@psu.edu]
#Project name: Social media influencers (SMI) talk about politics: predictors of individual's sharing intention towards SMI political advocacy tweets. 
#Independent Variables[9]: expertise, similarity, trustworthy, attractive, interactivity, parasocial, information quality, receptivity, involvement
#Dependent Variable[1]: sharing intention. 

###install the packages.   
install.packages("tidyverse")
install.packages("caret")

###load the library. 
library(tidyverse)
library(caret)

#import the data. 
smi <- read.csv(file = "regSMI.csv")
head(smi)

#explore the dataset. 
smiTib <- as_tibble(smi)
smiTib
summary(smiTib)

#plotting the relationship between each variable and dv. linear model fits.  
smiUntidy <- gather(smiTib, "Variable", "Value", -share)

ggplot(smiUntidy, aes(Value, share)) +
  facet_wrap(~ Variable, scales = "free_x") +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw()

#split the data into train and test set. 
set.seed(12345)
train_inds <- sample(1:nrow(smiTib),round(nrow(smiTib)*.7))
smiTib_train <- smiTib[train_inds,]
smiTib_test <- smiTib[-train_inds,]

#######################Fit the linear regression model: 
#Fit the linear model on train set. 
linear <- lm(share ~ expertise + similarity + trustworthy + attractive + interactivity + parasocial + infoq + recep + involve, data = smiTib_train)
summary(linear)

#Evaluate the linear model performance on test set. 
pred.linear <- predict(linear, newdata = smiTib_test)
data.frame(
  RMSE = RMSE(pred.linear, smiTib_test$share),
  Rsquare = R2(pred.linear, smiTib_test$share)
)
#RMSE:1.5628, Rsquare:0.3252
#The best model is the one with lowest prediction error(RMSE) and highest R-square (great model fit)


#In the following section, we'll use the {caret} package to fit ridge /lasso/elastic net model. 
#{caret}will automatically choose the best tuning parameter values, 
#compute the final model and evaluate the model performance using cross-validation techniques.

#First, set a grid range of lambda values from 10^(-3) to 10. 10 to the power of (minus)three
lambda <- 10^seq(-3, 1, length = 100)

#######################1. Compute the ridge regression: 
#Build the model on train set, using 10 fold cross validation. 
#For ridge regression, we need to specify alpha = 0. 
set.seed(123)
ridge <- train(
  share ~., data = smiTib_train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 0, lambda = lambda)
)
#Best tune lambda
ridge$bestTune$lambda
#Model coefficients
coef(ridge$finalModel, ridge$bestTune$lambda)
#Make predictions
predictions <- ridge %>% predict(smiTib_test)
#Model prediction performance
data.frame(
  RMSE = RMSE(predictions, smiTib_test$share),
  Rsquare = R2(predictions, smiTib_test$share)
)
##RMSE:1.5638, Rsquare 0.3377 


#######################2. Compute the lasso regression: 
#Build the model on train set, 10 fold cross-validation. 
#For Lasso regression, we need to specify alpha = 1. 
set.seed(123)
lasso <- train(
  share ~., data = smiTib_train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(alpha = 1, lambda = lambda)
)
#Best tune lambda
lasso$bestTune$lambda
#Model coefficients
coef(lasso$finalModel, lasso$bestTune$lambda)
#Make predictions
predictions <- lasso %>% predict(smiTib_test)
#Model prediction performance
data.frame(
  RMSE = RMSE(predictions, smiTib_test$share),
  Rsquare = R2(predictions, smiTib_test$share)
)

#RMSE: 1.5687, Rsquare:0.3259 

#######################3. Compute the Elastic Net Regression: 
#Build the model on train set, 10 fold cross validation. 
set.seed(123)
elastic <- train(
  share ~., data = smiTib_train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)
#Best tuning grid 
elastic$bestTune
#Model coefficients
coef(elastic$finalModel, elastic$bestTune$lambda)
#Make predictions
predictions <- elastic %>% predict(smiTib_test)
#Model prediction performance
data.frame(
  RMSE = RMSE(predictions, smiTib_test$share),
  Rsquare = R2(predictions, smiTib_test$share)
)

##RMSE: 1.5564, Rsquare:0.3382

###Conclusion: the elastic net model has the lowest RMSE. it's the best model. 
###This is the end of my tutorial.Thank you!






````````
####Solution 2: Use the {glmnet} package to compare the models. 

install.packages("glmnet")
library(glmnet)

###################################### Ridge Regression model

x = model.matrix(share~., smiTib_train)[,-10] # trim off the DV

y = smiTib_train %>%
  select(share) %>%
  unlist() %>%
  as.numeric()

#Compute the Best lambda value for ridge model 
set.seed(12345)
ridge_cv <- cv.glmnet(x, y, alpha = 0)

best_lambda <- ridge_cv$lambda.min
best_lambda

#Display the coefficients of the model 
best_ridge <- glmnet(x, y, alpha = 0, lambda = best_lambda)
coef(best_ridge)

#Make predictions on the test data
test.ridge <- model.matrix(share ~., smiTib_test)[,-1]
predictions.ridge <- best_ridge %>% predict(test.ridge) %>% as.vector()

#Model performance metrics
data.frame(
  RMSE = RMSE(predictions.ridge, smiTib_test$share),
  Rsquare = R2(predictions.ridge, smiTib_test$share)
)
##RMSE:1.6870, Rsquare:0.2352


###################################### Lasso Regression model 

# Find the best lambda using cross-validation
set.seed(123) 
lasso_cv <- cv.glmnet(x, y, alpha = 1)
# Display the best lambda value
lasso_cv$lambda.min  

# Fit the final model on the training data
best_lasso <- glmnet(x, y, alpha = 1, lambda = lasso_cv$lambda.min)
# Dsiplay regression coefficients
coef(best_lasso)

# Make predictions on the test data
test.lasso <- model.matrix(share ~., smiTib_test)[,-1]
predictions.lasso <- best_lasso %>% predict(test.lasso) %>% as.vector()


# Model performance metrics
data.frame(
  RMSE = RMSE(predictions.lasso, smiTib_test$share),
  Rsquare = R2(predictions.lasso, smiTib_test$share)
)

##RMSE: 1.7234, Rsquare:0.2327 


###################################### Elastic Net Regression model 

# Build the model using the training set
set.seed(123)
en.model <- train(
  share ~., data = smiTib_train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)
# Best tuning parameter (Alpha & Lambda value)
en.model$bestTune

# Coefficient of the final model. You need to specify the best lambda
coef(en.model$finalModel, en.model$bestTune$lambda) 

# Make predictions on the test data
en.test <- model.matrix(share ~., smiTib_test)[,-1]
en.predictions <- en.model %>% predict(smiTib_test)
# Model performance metrics
data.frame(
  RMSE = RMSE(en.predictions, smiTib_test$share),
  Rsquare = R2(en.predictions, smiTib_test$share)
)

##RMSE: 1.5564 Rsquare: 0.3382

#####Elastic Net model has the Lowest RMSE among linear, ridge and lasso. 





