library(readr)
library(dplyr)
library(caret)
library(ROSE)
library(xgboost)
library(randomForest)
library(nnet)
library(e1071)
library(klaR)
library(class)

setwd("/Users/user/Desktop/Fall Semester/CS699 A1 Data Mining - Jae Lee/Project")

# Read and preprocess the dataset
df <- read.csv("project_dataset.csv")

# Split the df into training and testing dataset and export each of them initial_train.csv, initial_test.csv with "o_bullied" columns as target
set.seed(42)
initial_trainIndex <- createDataPartition(df$o_bullied, p = .8, list = FALSE, times = 1)
initial_train <- df[initial_trainIndex, ]
initial_test <- df[-initial_trainIndex, ]
write.csv(initial_train, file = "initial_train.csv")
write.csv(initial_test, file = "initial_test.csv")

# Read the training dataset and preprocess it
target_column <- "o_bullied"
target_variable <- df[[target_column]]
df <- df[ , !(names(df) %in% target_column)]
data_standardized <- as.data.frame(scale(df))

# Feature selection
correlations <- cor(data_standardized, target_variable, use = "pairwise.complete.obs")
sorted_correlation_indices <- order(abs(correlations), decreasing = TRUE)
top_features <- names(data_standardized)[sorted_correlation_indices][1:160]
data_filtered <- data_standardized[, top_features, drop = FALSE]

# Combining filtered data with target variable
combined_data <- cbind(data_filtered, target_variable = target_variable)

# Class balancing
N_samples <- 1.55 * sum(table(combined_data$target_variable))
balanced_data <- ovun.sample(target_variable ~ ., data = combined_data, method = "over", N = N_samples)$data

# Splitting the dataset
set.seed(42)
trainIndex <- createDataPartition(balanced_data$target_variable, p = .8, list = FALSE, times = 1)
X_train <- balanced_data[trainIndex, -ncol(balanced_data)]
y_train <- balanced_data$target_variable[trainIndex]
X_test <- balanced_data[-trainIndex, -ncol(balanced_data)]
y_test <- balanced_data$target_variable[-trainIndex]
y_train <- factor(y_train)
y_test <- factor(y_test)

# Combine training datasets and export it as csv named preprocessed_data.csv
preprocessed_data <- cbind(X_train, y_train)
write.csv(preprocessed_data, file = "preprocessed_data.csv")

train_and_evaluate_random_forest <- function(X_train, y_train, X_test, y_test) {    
  # Train the Random Forest model
  model <- randomForest(y_train ~ ., data = X_train)
  pred <- predict(model, X_test)
  
  # Evaluate with confusion matrix
  return(confusionMatrix(factor(pred), factor(y_test)))
}

train_and_evaluate_logistic_regression <- function(X_train, y_train, X_test, y_test) {
  model <- multinom(y_train ~ ., data = X_train, MaxNWts = 10000, maxit = 1000)
  pred <- predict(model, X_test)
  return(confusionMatrix(factor(pred), factor(y_test)))
}

train_and_evaluate_svm <- function(X_train, y_train, X_test, y_test) {
  
  model <- svm(y_train ~ ., data = X_train, type = 'C-classification', kernel = "linear", cost = 10)
  pred <- predict(model, X_test)
  return(confusionMatrix(factor(pred), factor(y_test)))
}

train_and_evaluate_naive_bayes <- function(X_train, y_train, X_test, y_test) {
  # Remove near zero variance predictors
  nzv <- caret::nearZeroVar(X_train)
  X_train <- X_train[, -nzv]
  X_test <- X_test[, -nzv]

  # Train Naive Bayes model
  model <- e1071::naiveBayes(X_train, y_train)
  pred <- predict(model, X_test)

  # Evaluate with confusion matrix
  return(caret::confusionMatrix(factor(pred), factor(y_test)))
}


train_and_evaluate_knn <- function(X_train, y_train, X_test, y_test) {
  model <- knn(train = X_train, test = X_test, cl = y_train)
  return(confusionMatrix(factor(model), factor(y_test)))
}

train_and_evaluate_xgboost <- function(X_train, y_train, X_test, y_test) {
  # Convert factor to numeric (0 and 1) for XGBoost
  # Assuming your factor levels are something like c("No", "Yes")
  y_train_num <- as.numeric(y_train)-1   # Convert factors to 0 and 1
  y_test_num <- as.numeric(y_test)-1     # Convert factors to 0 and 1

  # Train the XGBoost model
  xgb_model <- xgboost(data = as.matrix(X_train), label = y_train_num, nrounds = 100, verbose = 0)

  # Make predictions on the test set
  xgb_pred <- predict(xgb_model, as.matrix(X_test))

  # Convert probabilities to class labels with a 0.5 threshold
  xgb_class_pred <- ifelse(xgb_pred > 0.5, 1, 0)

  # Convert predictions back to factor for confusion matrix
  xgb_class_pred_factor <- factor(xgb_class_pred, levels = c(0, 1), labels = levels(y_test))

  # Calculate and return the confusion matrix
  conf_matrix <- confusionMatrix(xgb_class_pred_factor, y_test)
  return(conf_matrix)
}


run_model <- function(model_name, X_train, y_train, X_test, y_test) {
  if (model_name == "LogisticRegression") {
    return(train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test))
  } else if (model_name == "SVM") {
    return(train_and_evaluate_svm(X_train, y_train, X_test, y_test))
  } else if (model_name == "RandomForest") {
    return(train_and_evaluate_random_forest(X_train, y_train, X_test, y_test))
  } else if (model_name == "NaiveBayes") {
    return(train_and_evaluate_naive_bayes(X_train, y_train, X_test, y_test))
  } else if (model_name == "KNN") {
    return(train_and_evaluate_knn(X_train, y_train, X_test, y_test))
  } else if (model_name == "XGBoost") {
    return(train_and_evaluate_xgboost(X_train, y_train, X_test, y_test))
  } else {
    stop("Unknown model")
  }
}



# Execute each model and print their confusion matrix
conf_matrix_random_forest <- run_model("RandomForest", X_train, y_train, X_test, y_test)
conf_matrix_SVM <- run_model("SVM", X_train, y_train, X_test, y_test)
conf_matrix_knn <- run_model("KNN", X_train, y_train, X_test, y_test)
conf_matrix_logistic_regression <- run_model("LogisticRegression", X_train, y_train, X_test, y_test)
conf_matrix_naive_bayes <- run_model("NaiveBayes", X_train, y_train, X_test, y_test)
conf_matrix_xgboost <- train_and_evaluate_xgboost(X_train, y_train, X_test, y_test)

# Print confusion matrices
print(conf_matrix_random_forest)
print(conf_matrix_SVM)
print(conf_matrix_naive_bayes)
print(conf_matrix_knn)
print(conf_matrix_logistic_regression)
print(conf_matrix_xgboost)



library(pROC)

# Create a reference confusion matrix
reference_matrix <- matrix(c(723, 33, 44, 733), nrow = 2, byrow = TRUE)
colnames(reference_matrix) <- c("0", "1")
rownames(reference_matrix) <- c("0", "1")

# Calculate TPR and FPR
TN <- reference_matrix[1, 1]
FP <- reference_matrix[1, 2]
FN <- reference_matrix[2, 1]
TP <- reference_matrix[2, 2]

# Calculate TPR and FPR
TPR <- TP / (TP + FN)
FPR <- FP / (FP + TN)

# Create a ROC curve
roc_curve <- roc(response = c(rep(0, TN + FP), rep(1, TP + FN)), predictor = c(rep(0, TN), rep(1, FP), rep(0, FN), rep(1, TP)))

# Plot the ROC curve
plot(roc_curve, print.auc = TRUE, auc.polygon = TRUE, grid = TRUE)

# Calculate the AUC
auc <- auc(roc_curve)
cat("AUC:", auc, "\n")
