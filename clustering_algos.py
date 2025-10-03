import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import cycle # Used for DBSCAN plotting

# --- 1. Data Loading and Robust Preprocessing ---
file_path = "Cleaned_Viral_Social_Media_Trends.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: The file 'Cleaned_Viral_Social_Media_Trends.csv' was not found.")
    exit()

# Feature Engineering
df['Total_Engagement'] = df['Likes'] + df['Shares'] + df['Comments']
df['Engagement_Rate'] = (df['Total_Engagement'] / df['Views']) * 100

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# Identify features and encode
categorical_cols = ['Platform', 'Hashtag', 'Content_Type', 'Region']
numerical_cols = ['Views', 'Likes', 'Shares', 'Comments', 'Total_Engagement', 'Engagement_Rate']

df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Scaling Numerical Features
scaler = StandardScaler()
df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])

# --- CRITICAL FIX: Define X_cluster by selecting only numerical types ---
X_cluster = df_processed.select_dtypes(include=np.number) 

# Remove the encoded target column and any Post ID columns
if 'Engagement_Level_Encoded' in X_cluster.columns:
    X_cluster = X_cluster.drop(columns=['Engagement_Level_Encoded'])
if 'Post_ID' in X_cluster.columns:
    X_cluster = X_cluster.drop(columns=['Post_ID'])
    
print("Data Preprocessing Complete. Running Algorithms and Visualizations...")

# --- Common Setup for Visualization ---
# Reduce dimensions to 2D using PCA for visualization scatter plots
pca = PCA(n_components=2, random_state=42)
principal_components = pca.fit_transform(X_cluster)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])


# -------------------------------------------------------------------
# --- 2. K-Means Clustering (Visualization 1: PCA Scatter Plot) ---
# -------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_cluster)
pca_df['KMeans_Cluster'] = kmeans_labels

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='KMeans_Cluster', data=pca_df, palette='Set1', s=60, alpha=0.7)
plt.title('1. K-Means Clusters Visualized with PCA (k=3)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# -------------------------------------------------------------------
# --- 3. Agglomerative Clustering (Visualization 2: Dendrogram) ---
# -------------------------------------------------------------------
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg_clustering.fit_predict(X_cluster)

# Calculate linkage matrix for dendrogram (using a sample due to large data size)
sample_data = X_cluster.sample(n=300, random_state=42)
linked_data = linkage(sample_data, method='ward')

plt.figure(figsize=(15, 6))
dendrogram(linked_data, orientation='top', distance_sort='descending', show_leaf_counts=True, truncate_mode='lastp', p=30)
plt.title('2. Agglomerative Clustering Dendrogram (Top 30 Merges)')
plt.xlabel('Sample Index or Cluster Size')
plt.ylabel('Distance')
plt.show()


# -------------------------------------------------------------------
# --- 4. DBSCAN Clustering (Visualization 3: PCA Scatter Plot with Noise) ---
# -------------------------------------------------------------------
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_cluster)

pca_df['DBSCAN_Cluster'] = dbscan_labels

plt.figure(figsize=(8, 6))
# DBSCAN plot requires custom colors to highlight noise (-1)
# Create a color map: noise (-1) will be black, clusters (0, 1, 2...) will be other colors
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
colors.insert(0, (0, 0, 0, 1)) # Black for noise (-1)

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black dots for noise with lower alpha
        marker_size = 6
        marker_color = 'k'
        marker_alpha = 0.3
    else:
        # Colored dots for clusters
        marker_size = 10
        marker_color = col
        marker_alpha = 0.8
    
    class_member_mask = (dbscan_labels == k)
    
    xy = principal_components[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=marker_color, 
             markeredgecolor='k', markersize=marker_size, alpha=marker_alpha)

plt.title('3. DBSCAN Clusters Visualized with PCA (Noise in Black)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
