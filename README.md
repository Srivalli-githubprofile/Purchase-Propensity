This project aims to predict whether an online shopper will make a purchase based on browsing behavior using a Random Forest Classifier. The dataset is preprocessed by scaling numerical features and encoding categorical features before training the model.

Dataset
The dataset used is online_shoppers_intention.csv.
It contains user session data, including features like page visit durations, bounce rates, visitor type, and special days.
The target variable (Revenue) indicates whether a purchase was made (1) or not (0).

Project Workflow
1. Load Dataset
The dataset is loaded from Google Drive.
Column names and data types are displayed for reference.
2. Data Preprocessing
Feature Selection: Selected numerical & categorical features relevant to predicting purchases.
Data Scaling: Standardized numeric features using StandardScaler().
One-Hot Encoding: Encoded categorical variables using OneHotEncoder().
3. Train-Test Split
Data is split into 70% training and 30% testing.
Stratified sampling ensures class balance.

Model Training
A Random Forest Classifier with n_estimators=100 and max_depth=10 is used.
The model is trained using balanced class weights to handle class imbalance.
