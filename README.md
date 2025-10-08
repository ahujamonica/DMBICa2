
# 📈 Social Media Trends Data Mining & Business Intelligence Project

## 🎯 Overview

This repository contains the complete implementation and analysis for a series of data mining and business intelligence (DMBI) experiments performed on a **Viral Social Media Trends Dataset**. The project focuses on predicting content engagement and discovering underlying patterns using both conceptual data warehousing models (Star/Snowflake) and practical machine learning techniques (Classification, Clustering, and Association Rule Mining) implemented in **Python**.

The primary goal is to transform raw social media metrics into actionable business insights.

## 📁 Dataset

The analysis is based on the `Viral_Social_Media_Trends.csv` dataset, featuring 5,000 posts across major platforms.

| Feature | Type | Description |
| `Platform`, `Region` | Categorical | Social media platform and geographical location. |
| `Views`, `Likes`, `Shares`, `Comments` | Numerical | Core engagement metrics. |
| `Engagement_Level` | Target | The final classification label (Low, Medium, High). |

## 🧪 Experiments Performed

The project covers six core experiments, demonstrating end-to-end DMBI proficiency.

### **1. Data Warehousing Schema Design (Conceptual)**

* **Star Schema:** Designed for fast query performance, featuring a central `FACT_POST_PERFORMANCE` table directly linked to denormalized dimensions (`DIM_PLATFORM`, `DIM_CONTENT`, etc.).
* **Snowflake Schema:** Designed for minimal data redundancy, normalizing the dimensions further (e.g., splitting Content into `DIM_CONTENT_TYPE` and `DIM_HASHTAG`).

### **2. Data Preprocessing & Feature Engineering**

* **Engineering:** Created essential features like `Total_Engagement` and `Engagement_Rate` (Total Engagement / Views).
* **Transformation:** Performed One-Hot Encoding on categorical features and used **StandardScaler** to normalize numerical metrics.

### **3. Exploratory Data Analysis (EDA) & Visualization**

* Generated visualizations (Heatmaps, Bar Plots, Box Plots) to analyze feature distributions and correlations.
* **Key Insight:** Confirmed a strong correlation between all explicit engagement metrics (`Likes`, `Shares`, `Comments`) and identified platform performance differences.

### **4. Classification Algorithms (Decision Tree & Naïve Bayes)**

**Goal:** Predict `Engagement_Level` (Low/Medium/High).

| Model | Implementation | Key Finding |
| :--- | :--- | :--- |
| **Decision Tree** | Implemented using `DecisionTreeClassifier` | Shows high feature importance for the engineered `Engagement_Rate` and scaled `Views`. |
| **Naïve Bayes** | Implemented using `GaussianNB` | Provides a baseline probabilistic prediction, showing competitive or slightly lower accuracy compared to the Decision Tree. |

### **5. Clustering Algorithms (k-means, Agglomerative, DBSCAN)**

**Goal:** Discover natural groupings and outliers in post performance.

| Model | Implementation | Key Finding |
| :--- | :--- | :--- |
| **k-means** | Partitioned data into 3 distinct clusters. | **Visualization:** PCA Scatter Plot shows clear separation of the high-engagement group. |
| **Agglomerative** | Hierarchical clustering. | **Visualization:** Dendrogram shows the merging hierarchy of the data points. |
| **DBSCAN** | Density-based clustering. | Identified the majority of the data points as **noise/outliers** (label -1), highlighting the uniqueness of most viral content. |

### **6. Association Rule Mining (Apriori Algorithm)**

**Goal:** Find strong co-occurrence patterns between categorical features.

* **Implementation:** Used `mlxtend`'s Apriori and Association Rules functions.
* **Key Insight:** Filtered rules by **Lift 1.0** to find positive associations. Strong rules confirm logical dependencies (e.g., specific content types appear on their corresponding platform) and discover hidden promotional links.

## 🛠️ Prerequisites

To run the Python scripts locally, you need the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn mlxtend scipy
````

## 🚀 Repository Structure

```
.
├── src/
│   ├── data_preprocessing.py # Experiment 2 (Preprocessing Code)
│   ├── classification.py     # Experiment 4 (DT/NB Code)
│   └── clustering.py         # Experiment 5 (KMeans/Agglomerative/DBSCAN Code)
├── data/
│   └── Viral_Social_Media_Trends.csv # Original Dataset
├── outputs/
│   ├── viz_kmeans_pca_scatter.png    # Cluster Visualization
│   └── viz_dt_feature_importance.png # Feature Importance Visualization
└── README.md
```
