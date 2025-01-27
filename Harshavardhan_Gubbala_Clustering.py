import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_and_preprocess_data(customers_path, transactions_path):
    customers_df = pd.read_csv(customers_path)
    transactions_df = pd.read_csv(transactions_path)
    trans_agg = transactions_df.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean', 'count'],
        'TransactionDate': ['min', 'max']
    }).reset_index()

    #Flatten multi-level column names
    trans_agg.columns = ['CustomerID', 'TotalSpent', 'AvgOrderValue', 
                         'TotalTransactions', 'FirstTransaction', 'LastTransaction']

    #Calculate customer lifetime
    trans_agg['FirstTransaction'] = pd.to_datetime(trans_agg['FirstTransaction'])
    trans_agg['LastTransaction'] = pd.to_datetime(trans_agg['LastTransaction'])
    trans_agg['CustomerLifetime'] = (trans_agg['LastTransaction'] - 
                                     trans_agg['FirstTransaction']).dt.days
    # Merge with customer data
    final_df = pd.merge(customers_df, trans_agg, on='CustomerID', how='inner')

    # One-hot encode categorical variables
    final_df = pd.get_dummies(final_df, columns=['Region'], drop_first=True)

    return final_df

def find_optimal_clusters(data, max_clusters=10):
    inertias, silhouette_scores, db_scores = [], [], []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        db_scores.append(davies_bouldin_score(data, kmeans.labels_))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')

    plt.subplot(1, 3, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')

    plt.subplot(1, 3, 3)
    plt.plot(range(2, max_clusters + 1), db_scores, marker='o')
    plt.title('Davies-Bouldin Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('DB Score')

    plt.tight_layout()
    plt.show()
    return inertias, silhouette_scores, db_scores

def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

def visualize_clusters(data, clusters):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.title('Clusters (PCA Reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()


customers_path = r'C:\Users\Harshavardhan rao\Downloads\Customers.csv'
transactions_path = r'C:\Users\Harshavardhan rao\Downloads\Transactions.csv'
final_df = load_and_preprocess_data(customers_path, transactions_path)

#Features for clustering
features = final_df.drop(columns=['CustomerID', 'CustomerName', 'SignupDate', 
                                      'FirstTransaction', 'LastTransaction'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
find_optimal_clusters(scaled_features)

#Number of clusters
n_clusters = 4
clusters, kmeans = perform_clustering(scaled_features, n_clusters)
visualize_clusters(scaled_features, clusters)
print(f'Davies-Bouldin Index: {davies_bouldin_score(scaled_features, clusters):.3f}')
print(f'Silhouette Score: {silhouette_score(scaled_features, clusters):.3f}')


