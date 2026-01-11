from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app) 

# Load the data and the AI vectors (the files you just downloaded)
# Note: Ensure these files are in the same folder as app.py
movies = pickle.load(open('movies_list.pkl', 'rb'))
vectors = pickle.load(open('movie_vectors.pkl', 'rb'))

# 1. UPDATED MODEL NAME: Must match your Colab model
model = SentenceTransformer('all-mpnet-base-v2')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # 2. Convert user text into a math vector
    query_vector = model.encode([user_query])
    
    # 3. Compare user vector with all movie vectors
    similarity = cosine_similarity(query_vector, vectors).flatten()
    
    # 4. Get top 5 matches
    # This sorts by highest similarity and takes the top 5
    top_indices = similarity.argsort()[-5:][::-1]
    
    results = []
    for i in top_indices:
        results.append({
            "id": int(movies.iloc[i].movie_id), # Ensure this matches your column name
            "title": movies.iloc[i].title,
            "score": round(float(similarity[i]) * 100, 2) # Added a similarity score!
        })

    return jsonify(results)

if __name__ == '__main__':
    # port 5000 is standard for local testing
    app.run(port=5000, debug=True)