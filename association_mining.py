import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. Data Loading and Preparation ---
file_path = "Cleaned_Viral_Social_Media_Trends.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: The file 'Cleaned_Viral_Social_Media_Trends.csv' was not found.")
    exit()

# Select only the categorical columns relevant for rule mining
df_apriori = df[['Platform', 'Hashtag', 'Content_Type', 'Region']].copy()
df_encoded = pd.get_dummies(df_apriori)

print("Data preparation for Apriori complete. Transaction matrix created.")

# --- 2. Apply Apriori Algorithm ---

# Find frequent itemsets: Min Support is kept at 1% (0.01)
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

print(f"\nFound {len(frequent_itemsets)} Frequent Itemsets (Min Support=0.01)")


# --- 3. Generate Association Rules (FINAL CORRECTED METHOD) ---

# Generate rules using LIFT as the metric, with a minimum threshold of 1.0.
# This finds all rules that show a positive correlation.
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

print(f"\nGenerated {len(rules)} Association Rules (Min Lift >= 1.0)")

# Display top 10 rules sorted by Lift
print("Top 10 Association Rules (Sorted by Lift):\n")
# Filter out the perfect rules (Lift = inf) that sometimes appear due to zero support
rules_final = rules[rules['lift'] < float('inf')].sort_values(by='lift', ascending=False)
print(rules_final.head(10).to_markdown(index=False))
