from gensim.models import Word2Vec
import pandas as pd

# Load the books data (adjust path as needed)
book = pd.read_csv(r'C://Users//bharg//Downloads//bookrec-flask-app//bookrec-flask-app//IntegrationTemplates//Books.csv')

# Prepare data for Word2Vec model training
book_titles = book['Book-Title'].values.tolist()
book_tokens = [title.split() for title in book_titles]

print(book_tokens)

# Train the Word2Vec model
word2vec_model = Word2Vec(sentences=book_tokens, vector_size=50, window=5, min_count=1, workers=4, sg=0)

# Save the Word2Vec model
word2vec_model.save('word2vec_model.model')
print("Word2Vec model saved.")
