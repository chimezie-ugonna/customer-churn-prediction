# Bank Customer Churn Prediction – Machine Learning Pipeline

This project implements a complete machine learning pipeline to predict customer churn in a bank. The objective is to identify customers likely to leave so the business team can proactively intervene and improve retention.

## Project Objective

Customer churn is a significant concern for the bank's credit card division. The goal of this project is to develop a predictive model that can identify at risk customers using historical data. By doing this, the bank can take early actions to reduce churn and retain valuable customers.

## Dataset Description

The dataset contains over 10,000 customer records, each with 23 features. These include:

- Demographic details (e.g., Age, Gender, Marital Status)
- Credit and transaction behavior (e.g., Credit Limit, Transaction Count, Utilization Ratio)
- Activity metrics (e.g., Inactive Months, Contact Frequency)

The target variable is `Attrition_Flag`, which indicates whether a customer has churned or not.

## Pipeline Overview

The entire machine learning pipeline is implemented in the `BankChurnProject.ipynb` notebook and includes:

1. **Data Exploration**
   - Review of dataset size, structure, missing values
   - Class imbalance check
   - Visualizations (e.g., churn by gender, education, card type)
   - Correlation heatmap to inspect numeric relationships

2. **Data Preprocessing**
   - Removal of unnecessary columns (e.g., CLIENTNUM, unused model artifacts)
   - Encoding of categorical variables using OneHotEncoding
   - Feature scaling with StandardScaler

3. **Feature Engineering**
   - Identification of key categorical and numerical variables
   - Extraction of useful features based on importance from model analysis

4. **Model Training**
   - Multiple classifiers trained and evaluated:
     - Logistic Regression
     - K-Nearest Neighbors
     - Decision Tree
     - Random Forest
     - Support Vector Machine
     - Naive Bayes
   - Evaluation metrics: accuracy, precision, recall, F1-score
   - Confusion matrix and ROC curve used for assessment

5. **Final Model Selection**
   - Random Forest performed best overall with high recall and precision
   - Feature importance identified key indicators of churn (e.g., Total_Trans_Ct, Total_Trans_Amt)

## How to Run the Project

### Option 1 – Google Colab

1. Clone this repository to your Google Drive or open the `.ipynb` notebook directly in Google Colab.
2. Upload the `BankChurners.csv` file when prompted.
3. Run the notebook cells sequentially.

### Option 2 – Run Locally with Jupyter (Untested)

This pipeline may also work locally using Jupyter Notebook, but it was not tested outside Google Colab.

To attempt local setup:

```bash
git clone https://github.com/chimezie-ugonna/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn

# Launch Jupyter
jupyter notebook
Then Open `BankChurnProject.ipynb` and run the cells in order.

## Dependencies

You will need the following Python libraries:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

These can be installed via:

pip install pandas numpy seaborn matplotlib scikit-learn
```

## Project Output

- Trained classification models with detailed performance comparison
- Confusion matrix and ROC curve for best performing model
- Feature importance analysis
- Business focused interpretation and final recommendations

## Notes

- This project was completed as part of a machine learning course.
- The pipeline is designed to be easy to understand and modify for similar classification tasks.
- The dataset used is included for reproducibility.
