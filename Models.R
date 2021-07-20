# Load libraries
library(tidyverse)
library(ggplot2)
library(caret)
library(mice)
library(VIM)
library(lattice)
library(ROSE)
library(pROC)
library(ROCR)
library(knitr)
library(kableExtra)
library(gbm)

###############################################################################
# METHOD 1: SYNTHETIC DATA GENERATION
###############################################################################

# We can see that we have an imbalanced data set with about 25% of positive 
# cases and 75% of negative cases. Even though The imbalance problem is not so 
# severe, we still need to take care of this because it can incur poor 
# predictive performance for the minority class which is the class we are 
# interested in.
# To deal with this problem, we will create a sample of synthetic data.

# The data generated from oversampling have expected amount of repeated 
# observations. Data generated from undersampling is deprived of important 
# information from the original data. This leads to inaccuracies in the 
# resulting performance.

# Split data into training and test set.
set.seed(1999)
train_index <- sort(sample(1:nrow(dat), nrow(dat)*(4/5)))

train_data <- dat[train_index, ]
test_data <- dat[-train_index, ]

# Frequency of the target variable in the training data set
table(train_data$target)

# Proportion of 2 classes in the training data set
prop.table(table(train_data$target))

train_balanced_both <- ovun.sample(target ~ ., data = train_data, 
                                   method = "both", p=0.5, 
                                   N=nrow(train_data), seed = 1)$data
train_data_balanced <- ROSE(target ~ ., data = train_data, seed = 1999)$data
prop.table(table(train_data_balanced$target))

# Models
# Stepwise with Linear Terms
# We will use the stepwise regresion model to select variables contribute 
# the most to the model.
full <- glm(target ~ ., data=train_data_balanced, family="binomial")
null <- glm(target ~ 1, data=train_data_balanced, family="binomial")

lr_step_mod <- step(null, list(lower=formula(null), upper=formula(full)), 
                    data=train_data_balanced, direction="both", trace=0)
options(scipen = 999)
summary(lr_step_mod)

# Save an object to a file
saveRDS(lr_step_mod, file = "lr_step_mod.rds")

# Read the model in
lr_step_mod <- readRDS(file = "lr_step_mod.rds")
lr_step_mod

# Logistic Regression Model
set.seed(1999)
lr_mod <- train(formula(lr_step_mod),
                data = train_data_balanced,
                method = "glm",
                family = "binomial",
                metric="ROC",
                trControl = trainControl(method="cv", number = 5,
                                         summaryFunction = twoClassSummary,
                                         classProbs = TRUE,
                                         savePredictions = TRUE),
                preProcess = c("center", "scale"))
lr_mod

# Save an object to a file
saveRDS(lr_mod, file = "lr_mod.rds")
# Read the model back
lr_mod <- readRDS(file = "lr_mod.rds")
lr_mod 

# TRAINING DATA SET
# Predict on training set
lr_mod_est_train <- predict(lr_mod, train_data_balanced)
# Confusion matrix
confusionMatrix(train_data_balanced$target, lr_mod_est_train, positive="Yes")

# TEST DATA SET
# Predicted value on test set
lr_mod_est_test <- predict(lr_mod, newdata = test_data)

# Confusion matrix
confusionMatrix(lr_mod_est_test, test_data$target, positive = "Yes")


# AUC
lr_mod_est_prob <-  predict(lr_mod, newdata = test_data, type="prob")
lr_test_auc <- roc(test_data$target, lr_mod_est_prob$Yes)$auc
lr_test_auc


# KNN Model
set.seed(1999)
knn_mod <- train(formula(lr_step_mod),
                 data = train_data_balanced,
                 method = "knn",
                 metric="ROC",
                 trControl = trainControl(method="cv", number = 5,
                                          summaryFunction = twoClassSummary,
                                          classProbs = TRUE,
                                          savePredictions = TRUE),
                 preProcess = c("center", "scale"),
                 tuneGrid=data.frame(k=1:10))
knn_mod

# Save an object to a file
saveRDS(knn_mod, file = "knn_mod.rds")

# Read the model back
knn_mod <- readRDS(file = "knn_mod.rds")
knn_mod

# TRAINING DATA SET
# Predict on training set
knn_mod_est_train <- predict(knn_mod, train_data_balanced)
# Confusion matrix
confusionMatrix(train_data_balanced$target, knn_mod_est_train, positive="Yes")

# TEST DATA SET
# Predicted value on test set
knn_mod_est_test <- predict(knn_mod, newdata = test_data)

# Confusion matrix
confusionMatrix(knn_mod_est_test, test_data$target, positive = "Yes")

# AUC
knn_mod_est_prob <-  predict(knn_mod, newdata = test_data, type="prob")
knn_test_auc <- roc(test_data$target, knn_mod_est_prob$Yes)$auc
knn_test_auc


# Random Forest Model
set.seed(1999)
rf_mod <- train(formula(lr_step_mod),
                data = train_data_balanced,
                method = "rf",
                metric="ROC",
                trControl = trainControl(method="cv", number = 5,
                                          summaryFunction = twoClassSummary,
                                          classProbs = TRUE,
                                          savePredictions = TRUE),
                tuneGrid=data.frame(mtry=1:9),
                ntree=100)
rf_mod

# Save an object to a file
saveRDS(rf_mod, file = "rf_mod.rds")
# Read RDS file
rf_mod <- readRDS(file = "rf_mod.rds")
rf_mod

varImp(rf_mod, scale=FALSE)

# TRAINING DATA SET
# Predict on training set
rf_mod_est_train <- predict(rf_mod, train_data_balanced)

# Confusion matrix
confusionMatrix(train_data_balanced$target, rf_mod_est_train, positive="Yes")

# TEST DATA SET
# Predicted value on test set
rf_mod_est_test <- predict(rf_mod, newdata = test_data)

# Confusion matrix
confusionMatrix(rf_mod_est_test, test_data$target, positive = "Yes")

# AUC
rf_mod_est_prob <-  predict(rf_mod, newdata = test_data, type="prob")
rf_test_auc <- roc(test_data$target, rf_mod_est_prob$Yes)$auc
rf_test_auc


# Boosting Model
set.seed(1999)
boost_mod <- train(formula(lr_step_mod),
                 data = train_data_balanced,
                 method = "gbm",
                 metric="ROC",
                 trControl = trainControl(method="cv", number = 5,
                                          summaryFunction = twoClassSummary,
                                          classProbs = TRUE,
                                          savePredictions = TRUE),
                 tuneGrid = expand.grid(n.trees=seq(100, 200, by=50),
                                        interaction.depth=1:7,
                                        shrinkage=0.1,
                                        n.minobsinnode=10),
                 verbose = F)

boost_mod

# Save an object to a file
saveRDS(boost_mod, file = "boost_mod.rds")
# Read the model back
boost_mod <- readRDS(file = "boost_mod.rds")
boost_mod

# TRAINING DATA SET
# Predict on training set
boost_mod_est_train <- predict(boost_mod, train_data_balanced)

# Confusion matrix
confusionMatrix(train_data_balanced$target, boost_mod_est_train, positive="Yes")

# TEST DATA SET
# Predicted value on test set
boost_mod_est_test <- predict(boost_mod, newdata = test_data)

# Confusion matrix
confusionMatrix(boost_mod_est_test, test_data$target, positive = "Yes")

# AUC
boost_mod_est_prob <-  predict(boost_mod, newdata = test_data, type="prob")
boost_test_auc <- roc(test_data$target, boost_mod_est_prob$Yes)$auc
boost_test_auc

varImp(boost_mod, scale=FALSE)


# LDA Model
set.seed(1999)
lda_mod <- train(formula(lr_step_mod),
                 data = train_data_balanced,
                 method = "lda",
                 metric="ROC",
                 trControl = trainControl(method="cv", number = 5,
                                          summaryFunction = twoClassSummary,
                                          classProbs = TRUE,
                                          savePredictions = TRUE),
                 preProcess = c("center", "scale"))

lda_mod

# Save an object to a file
saveRDS(lda_mod, file = "lda_mod.rds")
# Read the model back
lda_mod <- readRDS(file = "lda_mod.rds")
lda_mod

# TRAINING DATA SET
# Predict on training set
lda_mod_est_train <- predict(lda_mod, train_data_balanced)

# Confusion matrix
confusionMatrix(train_data_balanced$target, lda_mod_est_train, positive="Yes")

# TEST DATA SET
# Predicted value on test set
lda_mod_est_test <- predict(lda_mod, newdata = test_data)

# Confusion matrix
confusionMatrix(lda_mod_est_test, test_data$target, positive = "Yes")

# AUC
lda_mod_est_prob <-  predict(lda_mod, newdata = test_data, type="prob")
lda_test_auc <- roc(test_data$target, lda_mod_est_prob$Yes)$auc
lda_test_auc


# QDA Model
set.seed(1999)
qda_mod <- train(formula(lr_step_mod),
                 data = train_data_balanced,
                 method = "qda",
                 metric="ROC",
                 trControl = trainControl(method="cv", number = 5,
                                          summaryFunction = twoClassSummary,
                                          classProbs = TRUE,
                                          savePredictions = TRUE),
                 preProcess = c("center", "scale"))

qda_mod

# Save an object to a file
saveRDS(qda_mod, file = "qda_mod.rds")
# Read the model back
qda_mod <- readRDS(file = "qda_mod.rds")
qda_mod

# TRAINING DATA SET
# Predict on training set
qda_mod_est_train <- predict(qda_mod, train_data_balanced)

# Confusion matrix
confusionMatrix(train_data_balanced$target, qda_mod_est_train, positive="Yes")

# TEST DATA SET
# Predicted value on test set
qda_mod_est_test <- predict(qda_mod, newdata = test_data)

# Confusion matrix
confusionMatrix(qda_mod_est_test, test_data$target, positive = "Yes")

# AUC
qda_mod_est_prob <-  predict(qda_mod, newdata = test_data, type="prob")
qda_test_auc <- roc(test_data$target, qda_mod_est_prob$Yes)$auc
qda_test_auc

# Evaluation
results<-data.frame(
  Model = c("Logistic Regression", "KNN", "Random Forest", "Boosting", "LDA", "QDA"),
  ROC.Valid = c(lr_test_auc, knn_test_auc, rf_test_auc, boost_test_auc, lda_test_auc, qda_test_auc),
  Sens.Valid = c(sensitivity(lr_mod_est_test, test_data$target, positive="Yes"),
                 sensitivity(knn_mod_est_test, test_data$target, positive="Yes"),
                 sensitivity(rf_mod_est_test, test_data$target, positive="Yes"),
                 sensitivity(boost_mod_est_test, test_data$target, positive="Yes"),
                 sensitivity(lda_mod_est_test, test_data$target, positive="Yes"),
                 sensitivity(qda_mod_est_test, test_data$target, positive="Yes")),
  Spec.Valid = c(sensitivity(lr_mod_est_test, test_data$target),
                 sensitivity(knn_mod_est_test, test_data$target),
                 sensitivity(rf_mod_est_test, test_data$target),
                 sensitivity(boost_mod_est_test, test_data$target),
                 sensitivity(lda_mod_est_test, test_data$target),
                 sensitivity(qda_mod_est_test, test_data$target))
)

kable(results) %>%
  kable_styling()

# Plot ROC
roc_df <- list("Logistic Regression" = roc(test_data$target, lr_mod_est_prob$Yes),
               "KNN" = roc(test_data$target, knn_mod_est_prob$Yes), 
               "Random Forest" = roc(test_data$target, rf_mod_est_prob$Yes), 
               "Boosting" = roc(test_data$target, boost_mod_est_prob$Yes), 
               "LDA" = roc(test_data$target, lda_mod_est_prob$Yes), 
               "QDA" = roc(test_data$target, qda_mod_est_prob$Yes))

ggroc(roc_df, legacy.axes = TRUE) +
  labs(title = "Receiver Operating Characteristic (ROC)",
       y = "Sensitivity",
       x = "1 - Specificity") +
  scale_color_discrete(name = "Models")+
  theme_bw()

# Chosen model: Logistic Regression Model
lr_mod$finalModel
exp(lr_mod$finalModel$coefficients)
# odds ratios and 95% CI
exp(cbind(OR = lr_mod$finalModel$coefficients, confint(lr_mod$finalModel)))


###############################################################################
# METHOD 2: THRESHOLD ADJUSTMENT
###############################################################################

# Models
# Stepwise with Linear Terms
# We will still use the stepwise regression model to select variables 
# contribute the most to the model.
full <- glm(target ~ ., data=train_data, family="binomial")
null <- glm(target ~ 1, data=train_data, family="binomial")

lr_step_mod_imbal <- step(null, list(lower=formula(null), upper=formula(full)), 
                          data=train_data, direction="both", trace=0)
options(scipen = 999)
summary(lr_step_mod_imbal)

# Save an object to a file
saveRDS(lr_step_mod_imbal, file = "lr_step_mod_imbal.rds")
# Read the model in
lr_step_mod_imbal <- readRDS(file = "lr_step_mod_imbal.rds")
lr_step_mod_imbal


# Logistic Regression Model
set.seed(1999)
lr_mod_imbal <- train(formula(lr_step_mod_imbal),
                 data = train_data,
                 method = "glm",
                 family = "binomial",
                 metric="ROC",
                 trControl = trainControl(method="cv", number = 5,
                                          summaryFunction = twoClassSummary,
                                          classProbs = TRUE,
                                          savePredictions = TRUE),
                 preProcess = c("center", "scale"))
lr_mod_imbal

# Save an object to a file
saveRDS(lr_mod_imbal, file = "lr_mod_imbal.rds")
# Read the model back
lr_mod_imbal <- readRDS(file = "lr_mod_imbal.rds")
lr_mod_imbal

# TRAINING DATA SET
# Predict on training set
lr_mod_est_train_imbal <- predict(lr_mod_imbal, train_data)

# Confusion matrix
confusionMatrix(train_data$target, lr_mod_est_train_imbal, positive="Yes")

# TEST DATA SET
# Predicted value on test set
lr_mod_est_test_imbal <- predict(lr_mod_imbal, newdata = test_data)

# Confusion matrix
confusionMatrix(lr_mod_est_test_imbal, test_data$target, positive = "Yes")

# AUC
lr_mod_est_prob_imbal <-  predict(lr_mod_imbal, newdata = test_data, type="prob")
lr_test_auc_imbal <- roc(test_data$target, lr_mod_est_prob_imbal$Yes)$auc
lr_test_auc_imbal


# KNN Model
set.seed(1999)
knn_mod_imbal <- train(formula(lr_step_mod_imbal),
                       data = train_data,
                       method = "knn",
                       metric="ROC",
                       trControl = trainControl(method="cv", number = 5,
                                                summaryFunction = twoClassSummary,
                                                classProbs = TRUE,
                                                savePredictions = TRUE),
                       preProcess = c("center", "scale"),
                       tuneGrid=data.frame(k=1:10))
knn_mod_imbal

# Save an object to a file
saveRDS(knn_mod_imbal, file = "knn_mod_imbal.rds")
# Read the model back
knn_mod_imbal <- readRDS(file = "knn_mod_imbal.rds")
knn_mod_imbal

# TRAINING DATA SET
# Predict on training set
knn_mod_est_train_imbal <- predict(knn_mod_imbal, train_data)
# Confusion matrix
confusionMatrix(train_data$target, knn_mod_est_train_imbal, positive="Yes")

# TEST DATA SET
# Predicted value on test set
knn_mod_est_test_imbal <- predict(knn_mod_imbal, newdata = test_data)

# Confusion matrix
confusionMatrix(knn_mod_est_test_imbal, test_data$target, positive = "Yes")

# AUC
knn_mod_est_prob_imbal <-  predict(knn_mod_imbal, newdata = test_data, type="prob")
knn_test_auc_imbal <- roc(test_data$target, knn_mod_est_prob_imbal$Yes)$auc
knn_test_auc_imbal


# Random Forest Model
set.seed(1999)
rf_mod_imbal <- train(formula(lr_step_mod_imbal),
                      data = train_data,
                      method = "rf",
                      metric="ROC",
                      trControl = trainControl(method="cv", number = 5,
                                                summaryFunction = twoClassSummary,
                                                classProbs = TRUE,
                                                savePredictions = TRUE),
                      tuneGrid=data.frame(mtry=1:9),
                      ntree=100)
rf_mod_imbal

# Save an object to a file
saveRDS(rf_mod_imbal, file = "rf_mod_imbal.rds")
# Read RDS file
rf_mod_imbal <- readRDS(file = "rf_mod_imbal.rds")
rf_mod_imbal

varImp(rf_mod_imbal, scale=FALSE)

# TRAINING DATA SET
# Predict on training set
rf_mod_est_train_imbal <- predict(rf_mod_imbal, train_data)

# Confusion matrix
confusionMatrix(train_data$target, rf_mod_est_train_imbal, positive="Yes")

# TEST DATA SET
# Predicted value on test set
rf_mod_est_test_imbal <- predict(rf_mod_imbal, newdata = test_data)

# Confusion matrix
confusionMatrix(rf_mod_est_test_imbal, test_data$target, positive = "Yes")

# AUC
rf_mod_est_prob_imbal <-  predict(rf_mod_imbal, newdata = test_data, type="prob")
rf_test_auc_imbal <- roc(test_data$target, rf_mod_est_prob_imbal$Yes)$auc
rf_test_auc_imbal


# Boosting Model
set.seed(1999)
boost_mod_imbal <- train(formula(lr_step_mod_imbal),
                         data = train_data,
                         method = "gbm",
                         metric="ROC",
                         trControl = trainControl(method="cv", number = 5,
                                          summaryFunction = twoClassSummary,
                                          classProbs = TRUE,
                                          savePredictions = TRUE),
                         tuneGrid = expand.grid(n.trees=seq(100, 200, by=50),
                                                interaction.depth=1:7,
                                                shrinkage=0.1,
                                                n.minobsinnode=10),
                         verbose = F)

boost_mod_imbal

# Save an object to a file
saveRDS(boost_mod_imbal, file = "boost_mod_imbal.rds")
# Read the model back
boost_mod_imbal <- readRDS(file = "boost_mod_imbal.rds")
boost_mod_imbal

# TRAINING DATA SET
# Predict on training set
boost_mod_est_train_imbal <- predict(boost_mod_imbal, train_data)

# Confusion matrix
confusionMatrix(train_data$target, boost_mod_est_train_imbal, positive="Yes")

# TEST DATA SET
# Predicted value on test set
boost_mod_est_test_imbal <- predict(boost_mod_imbal, newdata = test_data)

# Confusion matrix
confusionMatrix(boost_mod_est_test_imbal, test_data$target, positive = "Yes")

# AUC
boost_mod_est_prob_imbal <-  predict(boost_mod_imbal, newdata = test_data, type="prob")
boost_test_auc_imbal <- roc(test_data$target, boost_mod_est_prob_imbal$Yes)$auc
boost_test_auc_imbal

varImp(boost_mod_imbal, scale=FALSE)


# LDA Model
set.seed(1999)
lda_mod_imbal <- train(formula(lr_step_mod_imbal),
                       data = train_data,
                       method = "lda",
                       metric="ROC",
                       trControl = trainControl(method="cv", number = 5,
                                                summaryFunction = twoClassSummary,
                                                classProbs = TRUE,
                                                savePredictions = TRUE),
                       preProcess = c("center", "scale"))

lda_mod_imbal

# Save an object to a file
saveRDS(lda_mod_imbal, file = "lda_mod_imbal.rds")
# Read the model back
lda_mod_imbal <- readRDS(file = "lda_mod_imbal.rds")
lda_mod_imbal

# TRAINING DATA SET
# Predict on training set
lda_mod_est_train_imbal <- predict(lda_mod_imbal, train_data)

# Confusion matrix
confusionMatrix(train_data$target, lda_mod_est_train_imbal, positive="Yes")

# TEST DATA SET
# Predicted value on test set
lda_mod_est_test_imbal <- predict(lda_mod_imbal, newdata = test_data)

# Confusion matrix
confusionMatrix(lda_mod_est_test_imbal, test_data$target, positive = "Yes")

# AUC
lda_mod_est_prob_imbal <-  predict(lda_mod_imbal, newdata = test_data, type="prob")
lda_test_auc_imbal <- roc(test_data$target, lda_mod_est_prob_imbal$Yes)$auc
lda_test_auc_imbal


# QDA Model
set.seed(1999)
qda_mod_imbal <- train(formula(lr_step_mod_imbal),
                       data = train_data,
                       method = "qda",
                       metric="ROC",
                       trControl = trainControl(method="cv", number = 5,
                                                summaryFunction = twoClassSummary,
                                                classProbs = TRUE,
                                                savePredictions = TRUE),
                       preProcess = c("center", "scale"))

qda_mod_imbal

# Save an object to a file
saveRDS(qda_mod_imbal, file = "qda_mod_imbal.rds")
# Read the model back
qda_mod_imbal <- readRDS(file = "qda_mod_imbal.rds")
qda_mod_imbal

# TRAINING DATA SET
# Predict on training set
qda_mod_est_train_imbal <- predict(qda_mod_imbal, train_data)

# Confusion matrix
confusionMatrix(train_data$target, qda_mod_est_train_imbal, positive="Yes")

# TEST DATA SET
# Predicted value on test set
qda_mod_est_test_imbal <- predict(qda_mod_imbal, newdata = test_data)

# Confusion matrix
confusionMatrix(qda_mod_est_test_imbal, test_data$target, positive = "Yes")

# AUC
qda_mod_est_prob_imbal <-  predict(qda_mod_imbal, newdata = test_data, type="prob")
qda_test_auc_imbal <- roc(test_data$target, qda_mod_est_prob_imbal$Yes)$auc
qda_test_auc_imbal

# Evaluation
results_imbal <- data.frame(
  Model = c("Logistic Regression", "KNN", "Random Forest", "Boosting", "LDA", "QDA"),
  ROC.Valid = c(lr_test_auc_imbal, knn_test_auc_imbal, rf_test_auc_imbal, 
                boost_test_auc_imbal, lda_test_auc_imbal, qda_test_auc_imbal),
  Sens.Valid = c(sensitivity(lr_mod_est_test_imbal, test_data$target, positive="Yes"),
                 sensitivity(knn_mod_est_test_imbal, test_data$target, positive="Yes"),
                 sensitivity(rf_mod_est_test_imbal, test_data$target, positive="Yes"),
                 sensitivity(boost_mod_est_test_imbal, test_data$target, positive="Yes"),
                 sensitivity(lda_mod_est_test_imbal, test_data$target, positive="Yes"),
                 sensitivity(qda_mod_est_test_imbal, test_data$target, positive="Yes")),
  Spec.Valid = c(sensitivity(lr_mod_est_test_imbal, test_data$target),
                 sensitivity(knn_mod_est_test_imbal, test_data$target),
                 sensitivity(rf_mod_est_test_imbal, test_data$target),
                 sensitivity(boost_mod_est_test_imbal, test_data$target),
                 sensitivity(lda_mod_est_test_imbal, test_data$target),
                 sensitivity(qda_mod_est_test_imbal, test_data$target))
)

kable(results_imbal) %>%
  kable_styling()

# Plot ROC
roc_df_imbal <- list("Logistic Regression" = roc(test_data$target, lr_mod_est_prob_imbal$Yes),
                     "KNN" = roc(test_data$target, knn_mod_est_prob_imbal$Yes), 
                     "Random Forest" = roc(test_data$target, rf_mod_est_prob_imbal$Yes), 
                     "Boosting" = roc(test_data$target, boost_mod_est_prob_imbal$Yes), 
                     "LDA" = roc(test_data$target, lda_mod_est_prob_imbal$Yes), 
                     "QDA" = roc(test_data$target, qda_mod_est_prob_imbal$Yes))

ggroc(roc_df_imbal, legacy.axes = TRUE) +
  labs(title = "Receiver Operating Characteristic (ROC)",
       y = "Sensitivity",
       x = "1 - Specificity") +
  scale_color_discrete(name = "Models")+
  theme_bw()

# Based on the ROC of the validation set, the logistic regression is 
# selected because of its highest ROC among 6 models. However, we need to 
# adjust the decision threshold since the data is imbalanced.

# Chosen model: Logistic Regression Model
coords(roc(test_data$target, lr_mod_est_prob_imbal$Yes),"best", ret="threshold")

lr_mod_est_resp_test_imbal <- predict(lr_step_mod_imbal, newdata = test_data, 
                                      type = "response")
preds <- as.factor(ifelse(lr_mod_est_resp_test_imbal > 0.2389457, "Yes", "No"))
confusionMatrix(preds, test_data$target, positive = "Yes")

lr_mod_imbal$finalModel

exp(lr_mod_imbal$finalModel$coefficients)

# odds ratios and 95% CI
exp(cbind(OR = lr_mod_imbal$finalModel$coefficients, 
          confint(lr_mod_imbal$finalModel)))


# Citation
citation("tidyverse") 
citation("ggplot2")
citation("caret")
citation("mice")
citation("VIM")
citation("lattice")
citation("ROSE")
citation("pROC")
citation("ROCR")
citation("knitr")
citation("kableExtra")
citation("gridExtra")
citation("gbm")
