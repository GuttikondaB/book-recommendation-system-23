#!/usr/bin/env python
# coding: utf-8

# Import required libraries
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import pickle
from flask_cors import CORS, cross_origin

# Initialize Flask app
app = Flask(__name__)
cors = CORS(app,resources={r'/ml/*': {'origins':['http://localhost:3000']}})

# Load pre-trained models and data (ensure these files are in the same directory)
kmeans_model = pickle.load(open(r'C://Users//bharg//Downloads//bookrec-flask-app//bookrec-flask-app//kmeans_model.pkl', 'rb'))
word2vec_model = Word2Vec.load('word2vec_model.model')
user_data = pd.read_csv(r'C://Users//bharg//Downloads//bookrec-flask-app//bookrec-flask-app//user_data.csv')  # Contains User-ID, avg_rating, Age, and Cluster
book_data = pd.read_csv(r'C://Users//bharg//Downloads//bookrec-flask-app//bookrec-flask-app//IntegrationTemplates/Books.csv')  # Contains ISBN, Book-Title, Book-Author, etc.
filtered_ratings = pd.read_csv(r'C://Users//bharg//Downloads//bookrec-flask-app//bookrec-flask-app//filtered_ratings.csv')  # Contains user ratings

# Define route for cluster-based recommendations
@app.route('/ml/cluster_recommendations/<int:user_id>', methods=['GET'])
def get_cluster_recommendations(user_id):
    """
    Recommends books to a user based on their cluster.
    """
    # Check if user exists in the user data
    user_cluster_data = user_data[user_data['User-ID'] == user_id]
    if user_cluster_data.empty:
        return jsonify({"error": "User not found"}), 404

    # Get the user's cluster
    user_cluster = user_cluster_data['Cluster'].values[0]

    # Find users in the same cluster
    cluster_users = user_data[user_data['Cluster'] == user_cluster]['User-ID']

    # Filter books rated by users in the same cluster
    cluster_books = filtered_ratings[filtered_ratings['User-ID'].isin(cluster_users)]
    top_books = cluster_books['ISBN'].value_counts().index[:10]  # Get top 10 books in the cluster

    # Fetch book details for recommendations
    recommended_books = book_data[book_data['ISBN'].isin(top_books)][['Book-Title', 'Book-Author', 'Year-Of-Publication']]
    return jsonify(recommended_books.to_dict(orient='records'))

# Define route for content-based recommendations using Word2Vec
@app.route('/ml//similar_books/<string:book_title>', methods=['GET'])
def get_similar_books(book_title):
    """
    Recommends books similar to a given book title using Word2Vec.
    """
    # Check if the book title exists in the Word2Vec vocabulary
    if book_title not in word2vec_model.wv:
        return jsonify({"error": "Book title not found"}), 404

    # Find similar books using the Word2Vec model
    similar_books = word2vec_model.wv.most_similar(book_title, topn=10)
    similar_titles = [book[0] for book in similar_books]

    # Fetch details for similar books
    book_details = book_data[book_data['Book-Title'].isin(similar_titles)][['Book-Title', 'Book-Author', 'Year-Of-Publication']]
    return jsonify(book_details.to_dict(orient='records'))

# Define route to test the API
@app.route('/', methods=['GET'])
def index():
    """
    Test endpoint to ensure the API is working.
    """
    return jsonify({"message": "Welcome to the Book Recommendation API!"})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
