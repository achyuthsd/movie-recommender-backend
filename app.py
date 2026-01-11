from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app) 

# Load the data and the AI vectors (L12 version)
movies = pickle.load(open('movies_list.pkl', 'rb'))
vectors = pickle.load(open('movie_vectors.pkl', 'rb'))

# 1. UPDATED MODEL NAME: Switched from mpnet to MiniLM-L12-v2
# This fits within Render's 512MB RAM limit
model = SentenceTransformer('all-MiniLM-L12-v2')

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
    top_indices = similarity.argsort()[-5:][::-1]
    
    results = []
    for i in top_indices:
        results.append({
            "id": int(movies.iloc[i].movie_id), 
            "title": movies.iloc[i].title,
            "score": round(float(similarity[i]) * 100, 2)
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=5000)