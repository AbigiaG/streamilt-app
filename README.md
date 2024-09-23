# streamilt-app
# Financial Inclusion in Africa: Machine Learning for Predicting Bank Account Ownership
# Project Overview 📊🌍
This project focuses on the Financial Inclusion in Africa dataset, provided by Zindi, which contains demographic information and financial service usage for around 33,600 individuals across East Africa. The goal is to build a machine learning (ML) model that predicts which individuals are most likely to have or use a bank account, contributing to efforts in financial inclusion.

Financial inclusion ensures that individuals and businesses have access to affordable financial products such as transactions, payments, savings, credit, and insurance, which meet their needs and are delivered in a responsible, sustainable way.

# Table of Contents
Dataset Overview
Key Features
Installation Instructions
Exploratory Data Analysis (EDA)
Model Training
Streamlit Application
Deployment
Conclusion
# Dataset Overview 📑
The dataset is sourced from the Zindi platform and includes demographic and financial service usage data for individuals. The dataset's primary goal is to help identify the factors influencing bank account ownership, aiding in financial inclusion efforts.

Number of Rows: 33,600
Region Covered: East Africa
Target Variable: Whether an individual has a bank account (Yes/No)
# Link to the dataset 📂

# Key Features
Country: The country the individual resides in.
Age of Respondent: The age of the individual.
Gender of Respondent: The gender of the respondent.
Marital Status: The marital status of the individual.
Level of Education: The highest level of education attained.
Type of Job: The type of employment the respondent is engaged in.
Location Type: Urban or rural area where the individual lives.
Cell Phone Access: Whether the respondent has access to a mobile phone.
Household Size: Number of people in the household.
Bank Account: The target column indicating if the individual owns a bank account.
Installation Instructions ⚙️
To replicate and run this project, follow the steps below:

Clone the repository:

git clone https://github.com/yourusername/financial-inclusion-africa.git
cd financial-inclusion-africa
Install necessary dependencies: Ensure you have Python 3.x installed, then run:

pip install -r requirements.txt
Launch the application:

streamlit run app.py
# Exploratory Data Analysis (EDA) 🔍
The EDA phase involved understanding the structure of the dataset, handling missing values, detecting outliers, and generating visual insights using a Pandas Profiling Report.

Steps:
General Information: Displayed the dataset's structure, data types, and overall statistics.
Missing Values: Handled missing and corrupted data through various imputation techniques.
Duplicates: Identified and removed duplicate entries.
Outliers: Detected and handled outliers using statistical methods.
Feature Encoding: Converted categorical variables to numerical formats suitable for machine learning models.
Pandas Profiling Report Example:

# pip install pandas-profiling
import pandas_profiling as pp
profile = pp.ProfileReport(df)
profile.to_file("output.html")
# Model Training 🧠
We trained a machine learning classifier to predict bank account ownership based on the demographic and financial features provided in the dataset. The following steps were taken:

# Data Preprocessing:
Feature scaling using StandardScaler.
One-hot encoding of categorical variables.
Model Selection:
We used models like Random Forest Classifier, Logistic Regression, and XGBoost.
Hyperparameter tuning was performed to improve model performance.
Evaluation Metrics:
Accuracy, Precision, Recall, and F1-Score were used to evaluate model performance.
Final Model: The best-performing model was a Random Forest Classifier, which achieved an accuracy of 85% on the test data.
# Streamlit Application 🎛️
We created a user-friendly Streamlit application that allows users to input features and receive predictions on bank account ownership. The app includes:

A form where users can input demographic details (age, gender, education, etc.).
A "Predict" button that uses the trained model to provide an instant prediction.
How to Run the App Locally
After installing the required packages, run the following command:

streamlit run app.py
# Deployment 🚀
The application was deployed on Streamlit Share, allowing for public access. The code was first pushed to a GitHub repository, and then the deployment was carried out using the Streamlit sharing platform.

# Steps for Deployment:
Create a GitHub repository and push the local project:

git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/financial-inclusion-africa.git
git push -u origin main
Deploy on Streamlit Share:

Log in to your Streamlit Share account.
Connect your GitHub repository and deploy the app from the repository.
# Conclusion 📈
The Financial Inclusion in Africa project successfully demonstrates how machine learning can be used to predict bank account ownership, contributing to increasing financial access for underrepresented populations. With the help of a robust EDA, feature engineering, and model tuning, we built a model that achieved promising results and deployed it as an interactive web application via Streamlit.

Repository Link
Feel free to explore the project on GitHub: GitHub Repository

# Streamlit Application Link
Access the live app: https://old-pens-cut.loca.lt/

Feel free to raise any issues or contribute to the project by submitting a pull request!

Author: [Abigia Edossa]
Contact: [abigia2000@gmail.com]
