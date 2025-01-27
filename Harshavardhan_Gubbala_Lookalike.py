import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np

customers = pd.read_csv('C:\Users\Harshavardhan rao\Downloads\Customers.csv')
products = pd.read_csv('C:\Users\Harshavardhan rao\Downloads\Products.csv')
transactions = pd.read_csv('C:\Users\Harshavardhan rao\Downloads\Transactions.csv')

merged_data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')

#Aggregated customer profiles
customer_profiles = merged_data.groupby('CustomerID').agg({
    'Category': lambda x: ' '.join(x),
    'Price': 'mean',
    'Quantity': 'sum'
}).reset_index()
customer_profiles['CombinedFeatures'] = customer_profiles['Category']

scaler = StandardScaler()
customer_profiles[['Price', 'Quantity']] = scaler.fit_transform(customer_profiles[['Price', 'Quantity']])

#TF-IDF vectorization for categorical data
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(customer_profiles['CombinedFeatures'])
numerical_features = customer_profiles[['Price', 'Quantity']].to_numpy()
combined_features = np.hstack((tfidf_matrix.toarray(), numerical_features))

#Lookalike Model
similarity_matrix = cosine_similarity(combined_features)

#Get top 3 similar customers for each customer
def get_top_similar(customers_df, similarity_matrix, top_n=3):
    lookalike_dict = {}
    for i, customer_id in enumerate(customers_df['CustomerID']):
        similar_indices = np.argsort(similarity_matrix[i])[::-1][1:top_n + 1]
        similar_customers = [
            (customers_df.iloc[j]['CustomerID'], similarity_matrix[i][j])
            for j in similar_indices
        ]
        lookalike_dict[customer_id] = similar_customers
    return lookalike_dict

lookalike_dict = get_top_similar(customers, similarity_matrix)

#Creating Lookalike sheet
lookalike_list = []
for customer_id, similars in lookalike_dict.items():
    lookalike_list.append({
        'CustomerID': customer_id,
        'Lookalikes': [f"{sim[0]}:{sim[1]:.2f}" for sim in similars]
    })

lookalike_df = pd.DataFrame(lookalike_list)
lookalike_df.to_csv('Lookalike.csv', index=False)

