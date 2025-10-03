import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# --- 1. Load Data ---
# Note: You must ensure this CSV file is uploaded to your Colab session or linked via Drive
file_path = "Cleaned_Viral_Social_Media_Trends.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: The file 'Cleaned_Viral_Social_Media_Trends.csv' was not found.")
    print("Please upload the file to your Colab session or verify the file path.")
    exit()

# --- 2. Feature Engineering ---
print("Starting Data Preprocessing...")
# Total Engagement = Likes + Shares + Comments
df['Total_Engagement'] = df['Likes'] + df['Shares'] + df['Comments']
# Engagement Rate = (Total Engagement / Views) * 100
df['Engagement_Rate'] = (df['Total_Engagement'] / df['Views']) * 100

# --- 3. Identify and Handle Missing Values ---
# Note: Since the provided data is 'Cleaned', we just drop any potential residuals.
df.dropna(inplace=True)

# --- 4. Identify Column Types for Encoding and Scaling ---
categorical_cols = ['Platform', 'Hashtag', 'Content_Type', 'Region']
numerical_cols = ['Views', 'Likes', 'Shares', 'Comments', 'Total_Engagement', 'Engagement_Rate']

# --- 5. Encoding Categorical Features ---

# One-Hot Encoding for input features (X)
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Label Encoding for the Target Variable (y)
le = LabelEncoder()
df_processed['Engagement_Level_Encoded'] = le.fit_transform(df['Engagement_Level'])

# --- 6. Scaling Numerical Features ---

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply standardization to the numerical columns
df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

# --- 7. Final Output and Save ---

# Define the final feature matrix (X) and target vector (y)
X = df_processed.drop(columns=['Post_ID', 'Engagement_Level', 'Engagement_Level_Encoded'])
y = df_processed['Engagement_Level_Encoded']

# Optional: Save processed data to CSV (recommended for subsequent Colab notebooks)
df_processed.to_csv('processed_social_media_data.csv', index=False)

print("\n--- Data Preprocessing Complete ---")
print(f"Shape of Final Processed Dataset: {df_processed.shape}")
print(f"Number of Features (X): {X.shape[1]}")
print(f"Processed data saved to 'processed_social_media_data.csv'")
print("\nSample of Scaled and Encoded Data:")
print(df_processed[['Views', 'Engagement_Rate', 'Platform_YouTube', 'Engagement_Level_Encoded']].head().to_markdown(index=False))
