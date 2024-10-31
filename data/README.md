# Data Folder

## Datasets

### Credit Card Fraud Detection Dataset

- **Source**: The Credit Card Fraud Detection dataset is available on Kaggle. 
- **Link**: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mishra5001/credit-card?resource=download)
- **Description**: This dataset contains detailed information about credit card applications and their associated fraud cases. It includes three separate CSV files with varying types of data related to credit card applications and historical data. The dataset is useful for analyzing credit card fraud, modeling, and predictive analytics.

#### Key Files:
1. **application_data.csv**: Contains data about each credit card application.
   - **Key Variables**:
     - `SK_ID_CURR`: Unique identifier for each application.
     - `NAME_CONTRACT_TYPE`: Type of loan or credit (e.g., Cash loan, Revolving credit).
     - `CODE_GENDER`: Gender of the applicant.
     - `FLAG_OWN_CAR`: Whether the applicant owns a car (1 = Yes, 0 = No).
     - `FLAG_OWN_REALTY`: Whether the applicant owns real estate (1 = Yes, 0 = No).
     - `AMT_INCOME_TOTAL`: Total annual income of the applicant.
     - `AMT_CREDIT`: Credit amount requested.
     - Additional demographic and financial variables.

2. **columns_description.csv**: Provides descriptions for the columns in the `application_data.csv` file. Useful for understanding the meaning and context of each variable.

3. **previous_application.csv**: Contains historical application data for credit card applications previously made by the applicants.
   - **Key Variables**:
     - `SK_ID_PREV`: Unique identifier for the previous application.
     - `SK_ID_CURR`: Unique identifier for the current application.
     - `NAME_CONTRACT_TYPE`: Type of loan or credit (e.g., Cash loan, Revolving credit).
     - `AMT_ANNUITY`: Annuity amount for the previous loan.
     - `AMT_APPLICATION`: Amount applied for in the previous loan.
     - `AMT_CREDIT`: Credit amount granted in the previous loan.
     - Additional historical application details.

### File Information

- **application_data.csv**: Contains information about the current credit card applications.
- **columns_description.csv**: Provides detailed descriptions of the columns in `application_data.csv`.
- **previous_application.csv**: Contains historical credit application data.
