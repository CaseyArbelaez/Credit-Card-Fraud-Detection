# EDA should only be conducted on portion of training data

#Import necessary libraries

library(dplyr)        # For data manipulation
library(readr)        # For reading CSV files
library(ggplot2)      # For data visualization
library(tidyr)        # For data tidying functions (e.g., gather)
library(purrr)        # For functional programming (e.g., map functions)

# Load the data
credit_data <- read_csv("../data/application_data.csv")

# Check for missing values
missing_values <- sapply(credit_data, function(x) sum(is.na(x)))
missing_values <- data.frame(Variable = names(missing_values), Missing = missing_values)
missing_values <- missing_values[missing_values$Missing > 0, ]  # Only show variables with missing values
missing_values <- missing_values[order(-missing_values$Missing), ]  # Sort by the number of missing values
print(missing_values)

# Summary statistics for numerical features
summary(select_if(credit_data, is.numeric))

# Select key numerical columns
key_numerical_vars <- c("AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "DAYS_EMPLOYED")

# Plot histograms for the key numerical variables
credit_data %>%
  select(all_of(key_numerical_vars)) %>%
  gather(key = "Variable", value = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  facet_wrap(~ Variable, scales = "free") +
  labs(title = "Distribution of Key Numerical Variables") +
  theme_minimal()

# Select key categorical columns
key_categorical_vars <- c("NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY")

# Plot bar charts for the key categorical variables
credit_data %>%
  select(all_of(key_categorical_vars)) %>%
  gather(key = "Variable", value = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_bar(fill = "steelblue", color = "black") +
  facet_wrap(~ Variable, scales = "free") +
  labs(title = "Distribution of Key Categorical Variables") +
  theme_minimal()

# PCA Analysis Section

# Remove non-numeric columns and target variable
numeric_data <- select_if(credit_data, is.numeric)

# Drop columns with a significant amount of missing values or perform imputation (if needed)
numeric_data <- numeric_data %>% na.omit()  # Simple approach: remove rows with missing data

# Standardize numeric features
numeric_data_scaled <- scale(numeric_data)


# Identify and remove zero-variance columns
non_constant_columns <- numeric_data[, apply(numeric_data, 2, var) != 0]

# Normalize the non-constant data (mean = 0, sd = 1)
numeric_data_scaled <- scale(non_constant_columns)

# Perform PCA
pca_model <- prcomp(numeric_data_scaled, center = TRUE, scale. = TRUE)

# Extract the explained variance for each component
explained_variance <- pca_model$sdev^2 / sum(pca_model$sdev^2)
cumulative_variance <- cumsum(explained_variance)

# Scree plot: Explained variance by each principal component
scree_plot <- ggplot(data.frame(PC = 1:length(explained_variance), Variance = explained_variance),
                     aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(x = "Principal Component", y = "Variance Explained", title = "Scree Plot") +
  theme_minimal()
print(scree_plot)


# Cumulative variance plot
cumulative_variance_plot <- ggplot(data.frame(PC = 1:length(cumulative_variance), CumulativeVariance = cumulative_variance),
                                   aes(x = PC, y = CumulativeVariance)) +
  geom_line(color = "steelblue") +
  geom_point(color = "steelblue") +
  labs(x = "Principal Component", y = "Cumulative Variance Explained", title = "Cumulative Explained Variance") +
  theme_minimal()
print(cumulative_variance_plot)

# UMAP Section was moved to a completely new file in the same folder...

# Data Sampling

# Define the columns to keep
categorical_cols <- c("CODE_GENDER", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR",                              "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",                       "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",                                          "WEEKDAY_APPR_PROCESS_START", "REG_REGION_NOT_LIVE_REGION")

numerical_cols <- c("AMT_ANNUITY", "AMT_CREDIT", "CNT_CHILDREN", "AMT_INCOME_TOTAL",                     "AMT_GOODS_PRICE", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_BIRTH", 
                    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", 
                    "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", 
                    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR", 
                    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", 
                    "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", 
                    "DAYS_LAST_PHONE_CHANGE")

target_col <- c("TARGET")

# Combine the categorical and numerical columns
columns_to_keep <- c(target_col, categorical_cols, numerical_cols)

# Use dplyr::select() without all_of()
selected_data <- credit_data %>%
  select(any_of(columns_to_keep))

# Calculate the number of missing values for each column
missing_values_summary <- colSums(is.na(selected_data))

# Print the missing values summary
print(missing_values_summary)

# Drop rows with any missing values
cleaned_data <- selected_data %>%
  drop_na()

# Print the dimensions of the cleaned dataset
print(dim(cleaned_data))

# Check for XNA values in each column
xna_counts <- apply(cleaned_data, 2, function(x) sum(x == "XNA", na.rm = TRUE))

# Print the counts of XNA values for each column
print(xna_counts)

# Create filter_data to exclude rows with specific values
filter_data <- cleaned_data %>%
  filter(
    CODE_GENDER != "XNA",
    !is.na(AMT_GOODS_PRICE),
    !is.na(AMT_ANNUITY),
    AMT_INCOME_TOTAL < 9e6
  )

table(filter_data$TARGET)

# Calculate and print the percentage of observations in the minority class
target_counts <- table(filter_data$TARGET)
minority_class_percentage <- (min(target_counts) / sum(target_counts)) * 100

# Print the percentage of the minority class
cat("Percentage of observations in the minority class:", round(minority_class_percentage, 2), "%\n")

# Load ggplot2 for visualization
library(ggplot2)

# Plot class imbalance
ggplot(filter_data, aes(x = as.factor(TARGET))) +
  geom_bar(fill = "lightgreen") +
  labs(title = "Class Distribution of TARGET Variable in Cleaned Dataset",
       x = "TARGET",
       y = "Count") +
  theme_minimal()


# Data sampling

# Set seed for reproducibility
set.seed(123)

target_size <- 30000

# Calculate the number of observations in each class
class_counts <- credit_data %>%
  group_by(TARGET) %>%
  summarize(count = n(), .groups = 'drop')

# Calculate the proportion of each class
class_proportions <- class_counts %>%
  mutate(proportion = count / sum(count))

# Determine the number of samples to take from each class
samples_per_class <- class_proportions %>%
  mutate(samples = round(target_size * proportion)) %>%
  select(TARGET, samples) %>%
  deframe()

# Function to sample data for each class
sample_class <- function(data, samples, class) {
  data %>%
    filter(TARGET == class) %>%
    slice_sample(n = samples, replace = FALSE)
}

# Apply the sampling function to each class
downsampled_data_list <- lapply(names(samples_per_class), function(class) {
  sample_class(filter_data, samples_per_class[[class]], class)
})

# Combine the downsampled data from each class
downsampled_data <- bind_rows(downsampled_data_list)

# Save the combined downsampled data to a CSV file
write_csv(downsampled_data, "../data/downsampled_data.csv")


# Calculate and print the percentage of observations in the minority class
target_counts <- table(downsampled_data$TARGET)


minority_class_percentage <- (min(target_counts) / sum(target_counts)) * 100

# Print the percentage of the minority class
cat("Percentage of observations in the minority class:", round(minority_class_percentage, 2), "%\n")

# Plot class imbalance
ggplot(downsampled_data, aes(x = as.factor(TARGET))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Class Distribution of TARGET Variable in Downsampled Data",
       x = "TARGET",
       y = "Count") +
  theme_minimal()


# Functions to visualize the distribution of our numerical predictors

# Function to plot and save histogram for a single variable with log transformation
plot_histogram_logs <- function(data, var_name, save_path = "../plots/") {
  # Apply log transformation
  transformed_var <- paste0("log(abs(", var_name, ") + 1)")
  
  # Create the plot
  p <- ggplot(data, aes_string(x = transformed_var)) +
    geom_histogram(bins = 30, fill = "steelblue", color = "black") +
    labs(
      title = paste("Histogram of", var_name, "(Log-transformed)"),
      x = paste("log(abs(", var_name, ") + 1)"),
      y = "Frequency"
    ) +
    theme_minimal()
  
  # Save the plot
  ggsave(filename = file.path(save_path, paste0("histogram_", var_name, "_log_transformed.png")), plot = p)
  
  # Return the plot for printing
  return(p)
}

# Function to plot and save histogram for a single variable
plot_histogram <- function(data, var_name, save_path = "../plots/") {
  # Create the plot
  p <- ggplot(data, aes_string(x = var_name)) +
    geom_histogram(bins = 30, fill = "steelblue", color = "black") +
    labs(
      title = paste("Histogram of", var_name),
      x = var_name,
      y = "Frequency"
    ) +
    theme_minimal()
  
  # Save the plot
  ggsave(filename = file.path(save_path, paste0("histogram_", var_name, ".png")), plot = p)
  
  # Return the plot for printing
  return(p)
}

# Function to plot and save histogram for a single variable with power transformation
plot_histogram_power <- function(data, var_name, power, save_path = "../plots/") {
  # Apply power transformation
  transformed_var <- paste0("abs(", var_name, ")^", power)
  
  # Create the plot
  p <- ggplot(data, aes_string(x = transformed_var)) +
    geom_histogram(bins = 30, fill = "steelblue", color = "black") +
    labs(
      title = paste("Histogram of", var_name, "(Power-transformed)"),
      x = paste("(", var_name, ")^", power),
      y = "Frequency"
    ) +
    theme_minimal()
  
  # Save the plot
  ggsave(filename = file.path(save_path, paste0("histogram_", var_name, "_power_transformed.png")), plot = p)
  
  # Return the plot for printing
  return(p)
}

# Create histograms for all numeric variables
numeric_vars <- names(downsampled_data)[sapply(downsampled_data, is.numeric)]

# Plot histograms for each numeric variable
plots <- lapply(numeric_vars, function(var) plot_histogram(downsampled_data, var))

# Print all plots
print(plots)


# Plot histograms for each numeric variable with log transformation
plots <- lapply(numeric_vars, function(var) plot_histogram_logs(downsampled_data, var))

# Print all plots
print(plots)

# Plot histograms for each numeric variable with log transformation
plots <- lapply(numeric_vars, function(var) plot_histogram_power(downsampled_data, var, 0.5))

# Print all plots
print(plots)




















