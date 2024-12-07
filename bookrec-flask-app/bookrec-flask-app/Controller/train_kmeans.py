import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle

# Load data (adjust paths as needed)
user = pd.read_csv(r'C://Users//bharg//Downloads//bookrec-flask-app//bookrec-flask-app//IntegrationTemplates//Users.csv')
book = pd.read_csv(r'C://Users//bharg//Downloads//bookrec-flask-app//bookrec-flask-app//IntegrationTemplates//Books.csv')
rating = pd.read_csv(r'C://Users//bharg//Downloads//bookrec-flask-app//bookrec-flask-app//IntegrationTemplates//Ratings.csv')

# Merge rating with book data for detailed ratings
rating = rating.merge(book, on='ISBN')
rating = rating.merge(user, on='User-ID')

# Filter ratings to retain users with more than 5 ratings
user_ratings_count = rating.groupby('User-ID').size()
filtered_users = user_ratings_count[user_ratings_count > 5].index

# Keep ratings only from filtered users
filtered_ratings = rating[rating['User-ID'].isin(filtered_users)]

# Compute user average ratings for clustering
user_preferences = filtered_ratings.groupby('User-ID')['Book-Rating'].mean().reset_index()
user_preferences.columns = ['User-ID', 'avg_rating']

# Add Age data and normalize
user_data = user_preferences.merge(user[['User-ID', 'Age']], on='User-ID').dropna()
user_data['Age'] = (user_data['Age'] - user_data['Age'].mean()) / user_data['Age'].std()

# Train the K-Means model
kmeans = KMeans(n_clusters=5, random_state=42)
user_data['Cluster'] = kmeans.fit_predict(user_data[['avg_rating', 'Age']])

# Save the K-Means model and user data with clusters
pickle.dump(kmeans, open('kmeans_model.pkl', 'wb'))
user_data.to_csv('user_data.csv', index=False)
print("K-Means model and user data saved as 'user_data.csv'.")

# Save the filtered ratings to a CSV file for reuse in the Flask API
filtered_ratings.to_csv('filtered_ratings.csv', index=False)
print("Filtered ratings saved as 'filtered_ratings.csv'.")
