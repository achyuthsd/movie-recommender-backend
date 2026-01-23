from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app) 


movies = pickle.load(open('movies_list.pkl', 'rb'))
vectors = pickle.load(open('movie_vectors.pkl', 'rb'))


model = SentenceTransformer('all-MiniLM-L12-v2')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    query_vector = model.encode([user_query])
    
    similarity = cosine_similarity(query_vector, vectors).flatten()
    
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
