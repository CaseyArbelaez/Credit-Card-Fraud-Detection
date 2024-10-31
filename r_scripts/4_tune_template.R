
# Import necessary libraries

library(dplyr)
library(ggplot2)
library(tune)
library(parsnip)
library(recipes)
library(doParallel)
library(png)
library(kableExtra)
library(htmltools)


# Define the number of cores for parallel processing
num_cores <- 6

# Create a cluster and register the parallel backend
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Perform cross-validation and fit the logistic regression models
logistic_cv_results_transformed <- logistic_transformed_workflow %>%
  fit_resamples(
    resamples = cv_folds,
    metrics = metric_set(accuracy, roc_auc)
  )

logistic_cv_results_regular <- logistic_regular_workflow %>%
  fit_resamples(
    resamples = cv_folds,
    metrics = metric_set(accuracy, roc_auc)
  )

# Stop the cluster
stopCluster(cl)

# Save cross-validation results
save(logistic_cv_results_transformed, file = "../results/logistic_cv_results_transformed.rda")
save(logistic_cv_results_regular, file = "../results/logistic_cv_results_regular.rda")

# Fit the final logistic regression models on the upsampled training data
final_logistic_fit_transformed <- logistic_transformed_workflow %>%
  fit(data = upsampled_data)

final_logistic_fit_regular <- logistic_regular_workflow %>%
  fit(data = upsampled_data)

# Predict on the test data
test_predictions_logistic_transformed <- final_logistic_fit_transformed %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data)

test_predictions_logistic_regular <- final_logistic_fit_regular %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data)

# Generate confusion matrices
confusion_matrix_logistic_transformed <- test_predictions_logistic_transformed %>%
  conf_mat(truth = TARGET, estimate = .pred_class)

confusion_matrix_logistic_regular <- test_predictions_logistic_regular %>%
  conf_mat(truth = TARGET, estimate = .pred_class)

# Convert confusion matrices to tibble for visualization
conf_matrix_tibble_logistic_transformed <- as_tibble(confusion_matrix_logistic_transformed$table)
conf_matrix_tibble_logistic_regular <- as_tibble(confusion_matrix_logistic_regular$table)

# Create and save heatmap-style confusion matrix for the transformed workflow
heatmap_logistic_transformed <- ggplot(conf_matrix_tibble_logistic_transformed, aes(x = Prediction, y = Truth, fill = n)) +
  geom_tile() +
  geom_text(aes(label = n), color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Logistic Regression (Transformed): Confusion Matrix",
    x = "Predicted Class",
    y = "True Class"
  ) +
  theme_minimal()

ggsave(filename = "../results/heatmap_logistic_transformed.png", plot = heatmap_logistic_transformed, width = 8, height = 6)

# Create and save heatmap-style confusion matrix for the regular workflow
heatmap_logistic_regular <- ggplot(conf_matrix_tibble_logistic_regular, aes(x = Prediction, y = Truth, fill = n)) +
  geom_tile() +
  geom_text(aes(label = n), color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Logistic Regression (Regular): Confusion Matrix",
    x = "Predicted Class",
    y = "True Class"
  ) +
  theme_minimal()

ggsave(filename = "../results/heatmap_logistic_regular.png", plot = heatmap_logistic_regular, width = 8, height = 6)

# View cross-validation results
logistic_cv_results_transformed %>%
  collect_metrics()

logistic_cv_results_regular %>%
  collect_metrics()

# Collect metrics from the transformed and regular logistic regression results
transformed_metrics <- logistic_cv_results_transformed %>%
  collect_metrics() %>%
  mutate(Model = "Transformed")

regular_metrics <- logistic_cv_results_regular %>%
  collect_metrics() %>%
  mutate(Model = "Regular")

# Combine the metrics into a single data frame
combined_metrics <- bind_rows(transformed_metrics, regular_metrics)

# Create a pretty table with the combined metrics
combined_metrics %>%
  select(Model, .metric, mean, std_err) %>%
  kable(col.names = c("Model", "Metric", "Mean", "Standard Error"),
        caption = "Logistic Regression Cross-Validation Results") %>%
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover", "condensed")) %>%
  save_kable(file = "../results/log_metric_table.html")

# Load the HTML file and display it
library(htmltools)

# Read the HTML file
table_html <- readLines("../results/log_metric_table.html")

# Display the table
browsable(HTML(paste(table_html, collapse = "\n")))


# Number of cores
num_cores <- 6

# Create a cluster
cl <- makeCluster(num_cores)

# Register the parallel backend
registerDoParallel(cl)

# Perform grid search with cross-validation on upsampled data for transformed recipe
knn_results_transformed <- knn_workflow_transformed %>%
  tune_grid(
    resamples = cv_folds,  # Use the cross-validation folds
    grid = knn_grid,       # Use the tuning grid with 10 levels
    metrics = metric_set(roc_auc, accuracy),  # Metrics for evaluation
    control = control_grid(save_pred = TRUE)
  )

# Save KNN results for transformed recipe
save(knn_results_transformed, file = "../results/knn_results_transformed.rda")

# Perform grid search with cross-validation on upsampled data for non-transformed recipe
knn_results_no_transformation <- knn_workflow_no_transformation %>%
  tune_grid(
    resamples = cv_folds,  # Use the cross-validation folds
    grid = knn_grid,       # Use the tuning grid with 10 levels
    metrics = metric_set(roc_auc, accuracy),  # Metrics for evaluation
    control = control_grid(save_pred = TRUE)
  )

# Save KNN results for non-transformed recipe
save(knn_results_no_transformation, file = "../results/knn_results_no_transformation.rda")

# Stop the cluster
stopCluster(cl)

# Finalize and fit the KNN model on the full training data for transformed recipe
best_knn_transformed <- select_best(knn_results_transformed, metric = "roc_auc")

# Create the final workflow with the best parameters
final_knn_workflow_transformed <- knn_workflow_transformed %>%
  finalize_workflow(best_knn_transformed)

# Fit the finalized model on the full training data
final_knn_fit_transformed <- final_knn_workflow_transformed %>%
  fit(data = upsampled_data)

# Predict on the test data
test_predictions_knn_transformed <- final_knn_fit_transformed %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data)  # Combine predictions with actual data for evaluation

# Calculate performance metrics on the test data
test_metrics_knn_transformed <- test_predictions_knn_transformed %>%
  metrics(truth = TARGET, estimate = .pred_class)

# Display the performance metrics
print(test_metrics_knn_transformed)

# Display the confusion matrix
confusion_matrix_knn_transformed <- test_predictions_knn_transformed %>%
  conf_mat(truth = TARGET, estimate = .pred_class)

# Convert confusion matrix to tibble for visualization
conf_matrix_tibble_knn_transformed <- as_tibble(confusion_matrix_knn_transformed$table)

# Create heatmap-style confusion matrix
heatmap_knn_transformed <- ggplot(conf_matrix_tibble_knn_transformed, aes(x = Prediction, y = Truth, fill = n)) +
  geom_tile() +
  geom_text(aes(label = n), color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "K-Nearest Neighbors: Confusion Matrix (Transformed Data)",
    x = "Predicted Class",
    y = "True Class"
  ) +
  theme_minimal()

# Save the plot
ggsave(filename = "../results/heatmap_knn_transformed.png", plot = heatmap_knn_transformed)

# Finalize and fit the KNN model on the full training data for non-transformed recipe
best_knn_no_transformation <- select_best(knn_results_no_transformation, metric = "roc_auc")

# Create the final workflow with the best parameters
final_knn_workflow_no_transformation <- knn_workflow_no_transformation %>%
  finalize_workflow(best_knn_no_transformation)

# Fit the finalized model on the full training data
final_knn_fit_no_transformation <- final_knn_workflow_no_transformation %>%
  fit(data = upsampled_data)

# Predict on the test data
test_predictions_knn_no_transformation <- final_knn_fit_no_transformation %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data)  # Combine predictions with actual data for evaluation

# Calculate performance metrics on the test data
test_metrics_knn_no_transformation <- test_predictions_knn_no_transformation %>%
  metrics(truth = TARGET, estimate = .pred_class)

# Display the performance metrics
print(test_metrics_knn_no_transformation)

# Display the confusion matrix
confusion_matrix_knn_no_transformation <- test_predictions_knn_no_transformation %>%
  conf_mat(truth = TARGET, estimate = .pred_class)

# Convert confusion matrix to tibble for visualization
conf_matrix_tibble_knn_no_transformation <- as_tibble(confusion_matrix_knn_no_transformation$table)

# Create heatmap-style confusion matrix
heatmap_knn_no_transformation <- ggplot(conf_matrix_tibble_knn_no_transformation, aes(x = Prediction, y = Truth, fill = n)) +
  geom_tile() +
  geom_text(aes(label = n), color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "K-Nearest Neighbors: Confusion Matrix (No Transformation)",
    x = "Predicted Class",
    y = "True Class"
  ) +
  theme_minimal()


# Save the plot
ggsave(filename = "../results/heatmap_knn_no_transformation.png", plot = heatmap_knn_no_transformation)


# Collect metrics from the KNN models with and without transformation
knn_metrics_transformed <- test_metrics_knn_transformed %>%
  mutate(Model = "Transformed")

knn_metrics_no_transformation <- test_metrics_knn_no_transformation %>%
  mutate(Model = "No Transformation")

# Combine the metrics into a single data frame
combined_knn_metrics <- bind_rows(knn_metrics_transformed, knn_metrics_no_transformation)

# Create a pretty table with the combined metrics
combined_knn_metrics %>%
  select(Model, .metric, .estimate) %>%
  kable(col.names = c("Model", "Metric", "Estimate"),
        caption = "K-Nearest Neighbors Performance Metrics") %>%
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover", "condensed")) %>%
  save_kable(file = "../results/knn_metric_table.html")

# Read the HTML file
table_html <- readLines("../results/knn_metric_table.html")

# Display the table
browsable(HTML(paste(table_html, collapse = "\n")))

# Number of cores
num_cores <- 6

# Create a cluster
cl <- makeCluster(num_cores)

# Register the parallel backend
registerDoParallel(cl)

# Perform grid search with cross-validation on upsampled data for transformed recipe
rf_results_transformed <- rf_workflow_transformed %>%
  tune_grid(
    resamples = cv_folds,  # Use the cross-validation folds
    grid = rf_grid_transformed,       # Use the tuning grid with 5 levels
    metrics = metric_set(roc_auc, accuracy),  # Metrics for evaluation
    control = control_grid(save_pred = TRUE)
  )

# Save RF results for transformed recipe
save(rf_results_transformed, file = "../results/rf_results_transformed.rda")

# Stop the cluster
stopCluster(cl)

best_rf_transformed <- select_best(rf_results_transformed, metric = "roc_auc")

# Create the final workflow with the best parameters
final_rf_workflow_transformed <- rf_workflow_transformed %>%
  finalize_workflow(best_rf_transformed)

# Fit the finalized model on the full training data
final_rf_fit_transformed <- final_rf_workflow_transformed %>%
  fit(data = upsampled_data)

# Predict on the test data
test_predictions_rf_transformed <- final_rf_fit_transformed %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data)  # Combine predictions with actual data for evaluation

# Calculate performance metrics on the test data
test_metrics_rf_transformed <- test_predictions_rf_transformed %>%
  metrics(truth = TARGET, estimate = .pred_class)

# Display the performance metrics
print(test_metrics_rf_transformed)

# Display the confusion matrix
confusion_matrix_rf_transformed <- test_predictions_rf_transformed %>%
  conf_mat(truth = TARGET, estimate = .pred_class)

# Convert confusion matrix to tibble for visualization
conf_matrix_tibble_rf_transformed <- as_tibble(confusion_matrix_rf_transformed$table)

# Create heatmap-style confusion matrix
heatmap_rf_transformed <- ggplot(conf_matrix_tibble_rf_transformed, aes(x = Prediction, y = Truth, fill = n)) +
  geom_tile() +
  geom_text(aes(label = n), color = "white", size = 6) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Random Forest: Confusion Matrix (Transformed Data)",
    x = "Predicted Class",
    y = "True Class"
  ) +
  theme_minimal()

# Save the plot
ggsave(filename = "../results/heatmap_rf_transformed.png", plot = heatmap_rf_transformed)

# Collect metrics from the Random Forest model with transformation
rf_metrics_transformed <- test_metrics_rf_transformed %>%
  mutate(Model = "Transformed")

# Combine the metrics into a single data frame
combined_rf_metrics <- rf_metrics_transformed

# Create a pretty table with the combined metrics
combined_rf_metrics %>%
  select(Model, .metric, .estimate) %>%
  kable(col.names = c("Model", "Metric", "Estimate"),
        caption = "Random Forest Performance Metrics") %>%
  kable_styling(full_width = FALSE, bootstrap_options = c("striped", "hover", "condensed")) %>%
  save_kable(file = "../results/rf_metric_table.html")

# Read the HTML file
table_html <- readLines("../results/rf_metric_table.html")

# Display the table
browsable(HTML(paste(table_html, collapse = "\n")))
