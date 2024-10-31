# Credit Card Fraud Detection

## Project Overview

This repository contains the final project for the course **Foundations of Data Science with R (STAT 359)**. The project involves analyzing and modeling credit card transaction data to detect fraudulent activities. The dataset used includes detailed credit card application data and historical application information, enabling a comprehensive analysis of credit card fraud.

## Data

The `data` folder contains the dataset used for analysis. It includes the following CSV files:

- **`application_data.csv`**: Contains information about each credit card application.
- **`columns_description.csv`**: Provides descriptions for the columns in `application_data.csv`.
- **`previous_application.csv`**: Contains historical application data for credit card applications.

For detailed descriptions of these files and their contents, refer to the `README.md` file located in the `data` folder.

## Plots

The `plots` folder contains all visualizations generated during the analysis. This includes histograms, bar plots, and other relevant plots used to explore and present the data.

## Scripts

The `scripts` folder contains R scripts for various stages of the project:

- **`initial_setup.R`**: Handles the initial setup and loading of required libraries. It ensures that the necessary packages are installed and loaded into the R session.
- **`eda.R`**: Performs exploratory data analysis (EDA) to uncover patterns and relationships in the data. This includes generating visualizations and summary statistics.
- **`recipes.R`**: Handles data preprocessing and feature engineering tasks such as data cleaning, transformation, and creating new features from existing data.
- **`tune_template.R`**: A template for model tuning and evaluation. This script includes code for hyperparameter tuning and comparing different models to select the best one.

## Reports

The `reports` folder includes documentation and reports related to the project:

- **`final_report.qmd`**: The comprehensive final report summarizing the entire project, including findings and model performance.
