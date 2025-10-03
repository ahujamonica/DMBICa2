import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Load Data and Minimal Feature Engineering ---
# Note: Ensure 'Viral_Social_Media_Trends.csv' is uploaded to your Colab session.
file_path = "Viral_Social_Media_Trends.csv"

try:
    df_eda = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: The file 'Viral_Social_Media_Trends.csv' was not found.")
    print("Please upload the CSV to your Colab session or verify the file path.")
    exit()

# Perform minimal feature engineering required for these visualizations
df_eda['Total_Engagement'] = df_eda['Likes'] + df_eda['Shares'] + df_eda['Comments']
df_eda['Engagement_Rate'] = (df_eda['Total_Engagement'] / df_eda['Views']) * 100

# Define columns for plotting
numerical_metrics = ['Views', 'Likes', 'Shares', 'Comments', 'Total_Engagement', 'Engagement_Rate']
target_column = 'Engagement_Level'

print("Data loaded and features engineered successfully.")


# --- 2. Visualization 1: Correlation Matrix (Heatmap) ---
# Shows the linear relationship between numerical engagement metrics
correlation_matrix = df_eda[numerical_metrics].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Engagement Metrics')
plt.tight_layout()
plt.show()
print("Visualization 1 (Correlation Heatmap) displayed.")


# --- 3. Visualization 2: Bar Plot of Average Engagement by Platform ---
# Shows which platform drives the highest average engagement
platform_engagement = df_eda.groupby('Platform')['Total_Engagement'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x='Platform', y='Total_Engagement', data=platform_engagement, palette='Set2')
plt.title('Average Total Engagement by Platform')
plt.xlabel('Platform')
plt.ylabel('Average Total Engagement')
plt.tight_layout()
plt.show()
print("Visualization 2 (Bar Plot) displayed.")


# --- 4. Visualization 3: Box Plot for Outlier Detection across Target Levels ---
# Uses log transformation on Views to handle the wide scale and visualize outliers
plt.figure(figsize=(10, 6))
# Using np.log1p (log(1+x)) to handle potential zero values gracefully
sns.boxplot(x=target_column, y=df_eda['Views'].apply(lambda x: np.log1p(x)), data=df_eda, palette='viridis')
plt.title('Log(Views) Distribution Across Engagement Levels')
plt.xlabel('Engagement Level')
plt.ylabel('Log(Views) (Outlier Detection)')
plt.tight_layout()
plt.show()
print("Visualization 3 (Box Plot) displayed.")
