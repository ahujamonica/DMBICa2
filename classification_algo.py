import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. Data Loading and Preprocessing ---
file_path = "Cleaned_Viral_Social_Media_Trends.csv"

try:
    # Load the cleaned dataset
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: The file 'Cleaned_Viral_Social_Media_Trends.csv' was not found.")
    print("Please ensure the CSV is uploaded to your Colab session.")
    exit()

# Feature Engineering (Ensuring total metrics are calculated)
df['Total_Engagement'] = df['Likes'] + df['Shares'] + df['Comments']
df['Engagement_Rate'] = (df['Total_Engagement'] / df['Views']) * 100

# Identify features and target
categorical_cols = ['Platform', 'Hashtag', 'Content_Type', 'Region']
numerical_cols = ['Views', 'Likes', 'Shares', 'Comments', 'Total_Engagement', 'Engagement_Rate']

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# Encoding Categorical Features
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Label Encoding for the Target Variable (Engagement_Level)
le = LabelEncoder()
df_processed['Engagement_Level_Encoded'] = le.fit_transform(df['Engagement_Level'])
class_names = le.classes_ # Store class names for visualization

# Scaling Numerical Features
scaler = StandardScaler()
df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

# --- CRITICAL FIX: Define final X (features) by dropping all non-numeric columns ---
# We select only columns of numerical types (float, int, uint8 from one-hot encoding)
X = df_processed.select_dtypes(include=np.number) 
# Now remove the encoded target column
X = X.drop(columns=['Engagement_Level_Encoded'])
# Define the target vector
y = df_processed['Engagement_Level_Encoded']

# Split dataset into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data Preprocessing and Split Complete.")

# -------------------------------------------------------------------
# --- 2. Implement and Evaluate Decision Tree ---
# -------------------------------------------------------------------
print("\n--- Decision Tree Classifier ---")
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train) # THIS WILL NOW WORK
y_pred_dt = dt_classifier.predict(X_test)

# Evaluation and Report
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt, target_names=class_names))

# Visualization: Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Reds', 
            xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# -------------------------------------------------------------------
# --- 3. Implement and Evaluate Naïve Bayes ---
# -------------------------------------------------------------------
print("\n--- Naïve Bayes Classifier ---")
# Use GaussianNB as features are scaled (Continuous)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

# Evaluation and Report
print(f"Naïve Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb, target_names=class_names))

# Visualization: Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.title('Naïve Bayes Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
