import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:\\Users\\MEHWISH\\Downloads\\customer_segmentation_data.csv")
print(df)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

print("\n--- Converting text columns to numbers (One-Hot Encoding) ---")
df = pd.get_dummies(df, drop_first=True) # 'drop_first=True' helps avoid multicollinearity.
print("Data after converting text columns:")
print(df.head())

# Identify all columns that are numerical and not potential ID columns.
features_for_clustering = [col for col in df.columns if df[col].dtype in ['int64', 'float64', 'uint8'] and 'ID' not in col.upper()]
# Select only these features for scaling.
X = df[features_for_clustering]
# Initialize the StandardScaler.
scaler = StandardScaler()         
X_scaled = scaler.fit_transform(X)                 #calculates mean/std and applies (x - mean) / std in one step.

# Convert the scaled data back to a DataFrame with original column names.
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("\n--- Data After Scaling (Looks different, but ready for ML!) ---")
print(df_scaled.head())


sse = []                         # To store the inertia for each K
k_range = range(1, 11)            # Test K from 1 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)                       # X is scaled data
    sse.append(kmeans.inertia_)

# Plotting the Elbow Method result
plt.figure(figsize=(10, 5))
plt.plot(k_range, sse, marker='o')
plt.title('Elbow Method to Find Optimal Number of Clusters (K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')


plt.grid(True)
plt.show()

print("This point suggests the optimal number of clusters (K).")


optimal_k = 4                                            #  Adjust this based on your plot analysis.
kmeans = KMeans(n_clusters=optimal_k, random_state=42,n_init=10)
df['Cluster'] = kmeans.fit_predict(X)                    # Assign clusters to the original DataFrame

print(f"\n--- Customers assigned to {optimal_k} clusters! ---")
print(df.head())
print("\n--- How many customers are in each cluster? ---")
print(df['Cluster'].value_counts())

# Select features to analyze for clusters.

features_for_analysis = [col for col in df.columns if col not in ['Cluster'] and col in features_for_clustering]

# Calculate the mean of each feature for each cluster.
# groupby().mean() is a vectorized Pandas operation.
cluster_means = df.groupby('Cluster')[features_for_analysis].mean()
print("\n--- Average Characteristics of Each Customer Cluster")
print(cluster_means)


plt.figure(figsize=(10, 5))
sns.scatterplot(x='income', y='spending_score', hue='Cluster', data=df, palette='viridis',alpha=0.7)
                
plt.title('Clusters by Income & Spending Score')
plt.xlabel('Income')
plt.ylabel('Spending score')
plt.legend(title='Customer Cluster')
plt.grid(True)
plt.show() 