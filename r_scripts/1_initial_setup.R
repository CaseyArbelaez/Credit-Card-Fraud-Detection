# Initial data checks & data splitting

library(dplyr)
library(readr)

credit_application_data <- read_csv("../data/application_data.csv")

summary(credit_data)

# Calculate and print the percentage of observations in the minority class
target_counts <- table(credit_data$TARGET)
minority_class_percentage <- (min(target_counts) / sum(target_counts)) * 100

# Print the percentage of the minority class
cat("Percentage of observations in the minority class:", round(minority_class_percentage, 2), "%\n")


















