import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from google.colab import drive  # Required for Google Colab


# Load Dataset


# Mount Google Drive
drive.mount('/content/drive')

# Load the dataset
DATA_PATH = "/content/drive/MyDrive/online_shoppers_intention.csv"  # Update path if needed
df = pd.read_csv(DATA_PATH)

# Display the first few rows
print(df.head())

# Display column names and data types
print("\nDataset Info:")
print(df.info())


# Prepare Features and Target Variable


print("\nPreparing features and target...")

# Target variable: Revenue (1 = Purchase, 0 = No Purchase)
df['target'] = df['Revenue'].astype(int)

# Selecting relevant features
features = [
    'Administrative', 'Administrative_Duration', 'Informational',
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
    'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'
]

X = df[features]
y = df['target']

# Train-Test Split


print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")

# Preprocessing Data (Scaling & Encoding)

print("\nPreprocessing data...")

# Separate numeric and categorical columns
num_features = ['Administrative', 'Administrative_Duration', 'Informational',
                'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

cat_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[num_features])
X_test_scaled = scaler.transform(X_test[num_features])

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = encoder.fit_transform(X_train[cat_features])
X_test_cat = encoder.transform(X_test[cat_features])

# Convert encoded categorical features to DataFrame
X_train_cat_df = pd.DataFrame(X_train_cat, columns=encoder.get_feature_names_out(cat_features))
X_test_cat_df = pd.DataFrame(X_test_cat, columns=encoder.get_feature_names_out(cat_features))

# Reset index before merging
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

# Merge scaled numeric and encoded categorical features
X_train_final = pd.concat([pd.DataFrame(X_train_scaled, columns=num_features), X_train_cat_df], axis=1)
X_test_final = pd.concat([pd.DataFrame(X_test_scaled, columns=num_features), X_test_cat_df], axis=1)

print("Data preprocessing completed!")


# Train Model (Random Forest)


print("\nTraining Random Forest model...")

model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
model.fit(X_train_final, y_train)

print("Model training completed!")


# Model Evaluation

print("\nEvaluating the model...")

# Make predictions
y_pred = model.predict(X_test_final)
y_proba = model.predict_proba(X_test_final)[:, 1]

# Print evaluation metrics
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
