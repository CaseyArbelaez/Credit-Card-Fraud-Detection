# Recipes and feature engineering

# Import necessary libraries

library(tidyverse)   # Includes ggplot2, readr, and other data manipulation functions
library(rsample)     # For splitting data into training and testing sets
library(ROSE)        # For oversampling and undersampling techniques
library(recipes)     # For creating and preprocessing recipes
library(workflows)   # For combining recipes and models into workflows
library(tune)        # For hyperparameter tuning
library(kknn)        # For K-Nearest Neighbors modeling
library(ranger)      # For Random Forest modeling


# Load the dataset
downsampled_data <- read_csv("../data/downsampled_data.csv")

# Convert target variable to factor
downsampled_data$TARGET <- as.factor(downsampled_data$TARGET)

# Perform the initial split on downsampled data
split <- initial_split(downsampled_data, prop = 0.8, strata = "TARGET")

# Create training and testing datasets
train_data <- training(split)
test_data <- testing(split)

# Upsample data
upsampled_data <- ovun.sample(TARGET ~ ., data = train_data, method = "over", N = 30000)$data

# Verify the new class distribution
table(upsampled_data$TARGET)

# Create folds
cv_folds <- vfold_cv(upsampled_data, v = 5, repeats = 1, strata = "TARGET")
```
```{r echo=FALSE}
# Plot class imbalance
ggplot(upsampled_data, aes(x = as.factor(TARGET))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Class Distribution of TARGET Variable in Upsampled Data",
       x = "TARGET",
       y = "Count") +
  theme_minimal()
```
```{r echo=FALSE}
# Calculate and print the percentage of observations in the minority class
target_counts <- table(upsampled_data$TARGET)
minority_class_percentage <- (min(target_counts) / sum(target_counts)) * 100

# Print the percentage of the minority class
cat("Percentage of observations in the minority class:", round(minority_class_percentage, 2), "%\n")


# Recipe with specified transformations
transformation_recipe <- recipe(TARGET ~ ., data = downsampled_data) %>%
  step_mutate(AMT_ANNUITY = log(abs(AMT_ANNUITY) + 1),
              AMT_CREDIT = log(abs(AMT_CREDIT) + 1),
              AMT_INCOME_TOTAL = log(abs(AMT_INCOME_TOTAL) + 1),
              AMT_GOODS_PRICE = log(abs(AMT_GOODS_PRICE) + 1),
              DAYS_EMPLOYED = log(abs(DAYS_EMPLOYED) + 1),
              DAYS_REGISTRATION = sqrt(abs(DAYS_REGISTRATION)),
              DAYS_BIRTH = sqrt(abs(DAYS_BIRTH))) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

transformation_recipe

# Recipe without transformations
no_transformation_recipe <- recipe(TARGET ~ ., data = downsampled_data) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

no_transformation_recipe


# Prepare and bake the transformed recipe
transformation_prep <- prep(transformation_recipe, training = downsampled_data)
transformed_data <- bake(transformation_prep, new_data = downsampled_data)

str(transformed_data)

# Prepare and bake the recipe without transformations
no_transformation_prep <- prep(no_transformation_recipe, training = downsampled_data)
no_transformation_data <- bake(no_transformation_prep, new_data = downsampled_data)

str(no_transformation_data)

# Define the logistic regression model
logistic_model <- logistic_reg() %>%
  set_engine("glm")

# Create a transformed workflow combining the recipe and the model
logistic_transformed_workflow <- workflow() %>%
  add_recipe(transformation_recipe) %>%
  add_model(logistic_model)

# Create a regular workflow combining the recipe and the model
logistic_regular_workflow <- workflow() %>%
  add_recipe(no_transformation_recipe) %>%
  add_model(logistic_model)

# Create the recipe for preprocessing with upsampled data
recipe_knn <- recipe(TARGET ~ ., data = upsampled_data) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%   # One-hot encode categorical variables
  step_zv(all_predictors()) %>%  # Remove zero variance predictors
  step_normalize(all_of(c("AMT_ANNUITY", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "DAYS_EMPLOYED", "DAYS_REGISTRATION"))) %>%  # Normalize numeric predictors
  step_center(all_of(c("AMT_ANNUITY", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "DAYS_EMPLOYED", "DAYS_REGISTRATION"))) %>%  # Center numeric predictors
  step_scale(all_of(c("AMT_ANNUITY", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "DAYS_EMPLOYED", "DAYS_REGISTRATION")))  # Scale numeric predictors

# Display the recipe
recipe_knn

# Recipe with specified transformations for KNN
transformation_recipe_knn <- recipe(TARGET ~ ., data = upsampled_data) %>%
  step_mutate(AMT_ANNUITY = log(abs(AMT_ANNUITY) + 1),
              AMT_CREDIT = log(abs(AMT_CREDIT) + 1),
              AMT_INCOME_TOTAL = log(abs(AMT_INCOME_TOTAL) + 1),
              AMT_GOODS_PRICE = log(abs(AMT_GOODS_PRICE) + 1),
              DAYS_EMPLOYED = log(abs(DAYS_EMPLOYED) + 1),
              DAYS_REGISTRATION = sqrt(abs(DAYS_REGISTRATION)),
              DAYS_BIRTH = sqrt(abs(DAYS_BIRTH))) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

# Display the transformed recipe
transformation_recipe_knn

# Recipe without transformations for KNN
no_transformation_recipe_knn <- recipe(TARGET ~ ., data = upsampled_data) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

# Display the non-transformed recipe
no_transformation_recipe_knn

# Prepare and bake the transformed recipe
prepped_recipe_knn_transformed <- transformation_recipe_knn %>%
  prep(training = upsampled_data, retain = TRUE)

preprocessed_train_data_knn_transformed <- bake(prepped_recipe_knn_transformed, new_data = upsampled_data)

# Inspect the structure and column names of the preprocessed data
str(preprocessed_train_data_knn_transformed)

# Define and apply the KNN recipe without transformations
no_transformation_recipe_knn <- recipe(TARGET ~ ., data = upsampled_data) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

# Prepare and bake the recipe without transformations
prepped_recipe_knn_no_transformation <- no_transformation_recipe_knn %>%
  prep(training = upsampled_data, retain = TRUE)

preprocessed_train_data_knn_no_transformation <- bake(prepped_recipe_knn_no_transformation, new_data = upsampled_data)

# Inspect the structure and column names of the preprocessed data
str(preprocessed_train_data_knn_no_transformation)

knn_spec <- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# Extract parameter set dials and create a grid for tuning with 10 levels
knn_params <- hardhat::extract_parameter_set_dials(knn_spec) %>%
  update(neighbors = neighbors(c(1, 20)))

knn_grid <- grid_regular(knn_params, levels = 10) 

# Create the KNN workflow for transformed data
knn_workflow_transformed <- workflow() %>%
  add_recipe(transformation_recipe_knn) %>%
  add_model(knn_spec)

# Create the KNN workflow for non-transformed data
knn_workflow_no_transformation <- workflow() %>%
  add_recipe(no_transformation_recipe_knn) %>%
  add_model(knn_spec)


# Recipe for random forest with transformations
transformation_recipe_rf <- recipe(TARGET ~ ., data = upsampled_data) %>%
  step_mutate(AMT_ANNUITY = log(abs(AMT_ANNUITY) + 1),
              AMT_CREDIT = log(abs(AMT_CREDIT) + 1),
              AMT_INCOME_TOTAL = log(abs(AMT_INCOME_TOTAL) + 1),
              AMT_GOODS_PRICE = log(abs(AMT_GOODS_PRICE) + 1),
              DAYS_EMPLOYED = log(abs(DAYS_EMPLOYED) + 1),
              DAYS_REGISTRATION = sqrt(abs(DAYS_REGISTRATION)),
              DAYS_BIRTH = sqrt(abs(DAYS_BIRTH))) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

# Display the recipe with transformations
transformation_recipe_rf

# Apply the recipe with transformations
prepped_recipe_rf_transformed <- transformation_recipe_rf %>%
  prep(training = upsampled_data, retain = TRUE)

preprocessed_train_data_rf_transformed <- bake(prepped_recipe_rf_transformed, new_data = upsampled_data)

# Inspect the structure and column names of the preprocessed data
str(preprocessed_train_data_rf_transformed)

# Define the random forest model specification
rf_spec_transformed <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Extract parameter set dials and create a grid for tuning
num_columns_rf_transformed <- ncol(preprocessed_train_data_rf_transformed)  # Number of columns after preprocessing
num_predictors_rf_transformed <- num_columns_rf_transformed - 1  # Subtract 1 for the TARGET column

rf_params_transformed <- hardhat::extract_parameter_set_dials(rf_spec_transformed) %>%
  update(mtry = mtry(c(1, floor(0.7 * num_predictors_rf_transformed)))) %>%
  update(min_n = min_n(c(1, 20)))  # Include min_n in the tuning parameters  

rf_grid_transformed <- grid_regular(rf_params_transformed, levels = 5)

# Create the random forest workflow with transformations
rf_workflow_transformed <- workflow() %>%
  add_recipe(transformation_recipe_rf) %>%
  add_model(rf_spec_transformed)










