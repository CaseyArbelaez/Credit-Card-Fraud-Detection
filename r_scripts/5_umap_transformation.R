# Load necessary packages
library(tidyverse)
library(tidymodels)
library(recipes)
library(parsnip)
library(glmnet)
library(ROSE)
library(doParallel)
library(workflows)
library(tune)
library(furrr)
library(yardstick)
library(here)
library(rsample)
library(umap)
library(rgl)

credit_application_data <- read_csv("../data/application_data.csv")

# Define the recipe for one-hot encoding
one_hot_recipe <- recipe(TARGET ~ ., data = credit_application_data) %>%
  step_dummy(all_nominal(), -all_outcomes())  # Apply one-hot encoding to all categorical columns

# Prepare the recipe
one_hot_prep <- prep(one_hot_recipe, training = credit_application_data)

# Apply the recipe to transform the data
one_hot_encoded_data <- bake(one_hot_prep, new_data = credit_application_data)

# Inspect the transformed data
str(one_hot_encoded_data)
table(credit_application_data$TARGET)

# Count the number of missing values (NAs) per column
na_count <- colSums(is.na(one_hot_encoded_data))

# Identify columns with more than 10,000 NAs
columns_to_drop <- names(na_count[na_count > 100000])

# Drop these columns from the dataset
cleaned_data <- one_hot_encoded_data %>%
  select(-all_of(columns_to_drop))

# Drop all rows with missing values
cleaned_data <- cleaned_data %>%
  drop_na()

# Check the dimensions of the cleaned data
dim(cleaned_data)

# Optionally, display the dropped columns
columns_to_drop

# Step 2: Calculate the number of observations for each class
class_counts <- cleaned_data %>%
  group_by(TARGET) %>%
  summarize(count = n(), .groups = 'drop')

# Calculate the proportion of each class
class_proportions <- class_counts %>%
  mutate(proportion = count / sum(count))

# Determine the number of samples to take from each class
samples_per_class <- class_proportions %>%
  mutate(samples = round(15000 * proportion)) %>%
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
  sample_class(cleaned_data, samples_per_class[[class]], class)
})

# Combine the downsampled data from each class
downsampled_data <- bind_rows(downsampled_data_list)

# Save the combined downsampled data to a CSV file
write_csv(downsampled_data, "../data/downsampled_data.csv")

# Verify the new class distribution
table(downsampled_data$TARGET)

colSums(is.na(downsampled_data))

# Convert the TARGET variable to a factor
finalized_dataset <- finalized_dataset %>%
  mutate(TARGET = as.factor(TARGET))

downsampled_data <- downsampled_data %>%
  mutate(TARGET = as.factor(TARGET))

# Select numeric columns (excluding target variable)
numeric_data <- select(downsampled_data, -TARGET, -SK_ID_CURR)

# Normalize the numeric data (mean = 0, sd = 1)
numeric_data <- scale(numeric_data)

# Create a cluster with 6 cores
num_cores <- 6
cl <- makeCluster(num_cores)

# Register the parallel backend
registerDoParallel(cl)

# Perform UMAP with specific parameters
umap_result_1 <- umap(
  numeric_data, 
  n_components = 2,   # 3D UMAP
  n_neighbors = 30    # Number of neighbors
)

# Stop the cluster
stopCluster(cl)

# Create a data frame for 2D plotting
umap_data_1 <- as_tibble(umap_result_1$layout) %>%
  rename(UMAP1 = V1, UMAP2 = V2) %>%
  mutate(TARGET = downsampled_data$TARGET)  # Use downsampled_data for TARGET

# Plot UMAP results in 2D using ggplot2
ggplot(umap_data_1, aes(x = UMAP1, y = UMAP2, color = TARGET)) +
  geom_point(alpha = 0.7) +
  labs(title = "UMAP 2D Visualization",
       x = "UMAP Dimension 1",
       y = "UMAP Dimension 2") +
  theme_minimal() +
  scale_color_manual(values = c("blue", "red"))  # Customize colors if needed

# Create a data frame for 3D plotting
umap_data_1 <- as_tibble(umap_result_1$layout) %>%
  rename(UMAP1 = V1, UMAP2 = V2, UMAP3 = V3) %>%
  mutate(TARGET = downsampled_data$TARGET)  # Use downsampled_data for TARGET

# Open a new 3D device
open3d()

# Plot 3D scatter plot
plot3d(
  x = umap_data_1$UMAP1,
  y = umap_data_1$UMAP2,
  z = umap_data_1$UMAP3,
  col = as.numeric(umap_data_1$TARGET),  # Color by TARGET
  type = "s",  # Type of plot, "s" for spheres
  size = 3,    # Size of the points
  xlab = "UMAP Dimension 1",
  ylab = "UMAP Dimension 2",
  zlab = "UMAP Dimension 3",
  main = "UMAP 3D Visualization"
)

# Add a legend if needed
legend3d("topright", legend = levels(umap_data_1$TARGET), col = 1:length(levels(umap_data_1$TARGET)), pch = 16)

# Define parameter combinations for UMAP
param_combinations <- expand.grid(
  n_neighbors = c(5, 10, 15, 30, 50, 75, 100),
  min_dist = c(0.1, 0.2, 0.3, 0.4, 0.5),
  spread = c(0.5, 1, 1.5, 2, 3),
  stringsAsFactors = FALSE
)

# Initialize a list to store plots
plots <- list()

# Create a cluster with 6 cores
num_cores <- 6
cl <- makeCluster(num_cores)

# Register the parallel backend
registerDoParallel(cl)

# Loop through each combination of parameters
for (i in seq_len(nrow(param_combinations))) {
  params <- param_combinations[i, ]
  
  # Perform UMAP with current parameters
  umap_result <- umap(
    numeric_data,
    n_neighbors = params$n_neighbors,
    min_dist = params$min_dist,
    spread = params$spread
  )
  
  # Create a data frame for plotting
  umap_data <- as_tibble(umap_result$layout) %>%
    rename(UMAP1 = V1, UMAP2 = V2) %>%
    mutate(TARGET = downsampled_data$TARGET)  # Use downsampled_data for TARGET
  
  # Create a plot
  plot <- ggplot(umap_data, aes(x = UMAP1, y = UMAP2, color = TARGET)) +
    geom_point() +
    labs(
      title = paste("UMAP n_neighbors =", params$n_neighbors, 
                    ", min_dist =", params$min_dist, 
                    ", spread =", params$spread),
      x = "UMAP Dimension 1", y = "UMAP Dimension 2"
    ) +
    theme_minimal()
  
  # Save the plot in the list
  plots[[i]] <- plot
  
  # Save the plot to the "../plots" folder
  ggsave(filename = paste0("../plots/umap_plot_", i, ".png"), plot = plot)
}

stopCluster(cl)

# Print the plots one by one
for (i in seq_along(plots)) {
  print(plots[[i]])
}


