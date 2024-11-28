from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Define global keywords dictionary
DISASTER_KEYWORDS = {
    "earthquake": ["earthquake", "tremor", "seismic", "magnitude", "richter"],
    "flood": ["flood", "rainfall", "water", "deluge", "submerged"],
    "wildfire": ["fire", "wildfire", "bush fire", "forest fire", "flames"]
}

# 1. Load Pre-Trained Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Disaster Messages with sources
sentences = [
    {"message": "Flood in the northern region has affected thousands.", "source": "Reuters"},
    {"message": "Wildfire spreading rapidly near the city.", "source": "AP News"},
    {"message": "Relief camps set up for earthquake victims.", "source": "BBC"},
    {"message": "Heavy rainfall predicted for the coming week.", "source": "Weather Channel"},
    {"message": "Volunteers needed for flood relief efforts.", "source": "Local News"},
    {"message": "6.2 magnitude earthquake hits coastal region.", "source": "USGS"},
    {"message": "Mild tremors felt in metropolitan area.", "source": "Local News"},
    {"message": "Forest fire threatens residential areas.", "source": "CNN"},
    {"message": "Flash floods wash away bridges in valley.", "source": "AP"},
    {"message": "Earthquake of magnitude 4.5 recorded.", "source": "Reuters"}
]

# Extract just messages for embedding
messages = [item["message"] for item in sentences]

# 3. Generate Sentence Embeddings
embeddings = model.encode(messages)

# 5. Disaster Type Classification
# Updated labels: 0 - Flood, 1 - Wildfire, 2 - Earthquake
labels = [0, 1, 2, 0, 0, 2, 2, 1, 0, 2]  # Updated with new training data

# Train Classifier
clf = RandomForestClassifier()
clf.fit(embeddings, labels)  # Using all data for demo purposes

# 6. Clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)

def predict_disaster_type(message, embedding, similarity_scores):
    # Use global keywords dictionary
    message_lower = message.lower()
    
    # First check keywords
    for disaster_type, terms in DISASTER_KEYWORDS.items():
        if any(term in message_lower for term in terms):
            if disaster_type == "earthquake":
                return 2
            elif disaster_type == "flood":
                return 0
            elif disaster_type == "wildfire":
                return 1
    
    # If no keywords found and similarity score is very low, return -1 (Not Disaster Related)
    max_similarity = float(np.max(similarity_scores))
    if max_similarity < 0.2:  # Threshold for non-disaster content
        return -1
        
    # If score is higher but no keywords, use the classifier
    return clf.predict(embedding)[0]

# API Routes
@app.route('/')
def home():
    return jsonify({
        "message": "Disaster Analysis API is running",
        "available_endpoints": [
            "/analyze (POST)",
            "/get_similarities (GET)"
        ]
    })

@app.route('/analyze', methods=['POST'])
def analyze_message():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Please provide a message in the request body"}), 400

        message = data['message']
        source = data.get('source', 'Unknown Source')
        embedding = model.encode([message])
        
        # Calculate similarities
        similarity_scores = cosine_similarity(embedding, embeddings)
        
        # Predict cluster
        cluster = kmeans.predict(embedding)
        
        # Use enhanced disaster type prediction with similarity scores
        disaster_type = predict_disaster_type(message, embedding, similarity_scores)
        
        # Map disaster type to label with detailed descriptions
        disaster_mapping = {
            -1: {
                "type": "Not Disaster Related",
                "description": "This message does not appear to be related to any natural disaster",
                "severity": "None"
            },
            0: {
                "type": "Flood",
                "description": "Events involving water-related disasters including floods, heavy rainfall, and water damage",
                "severity": "High" if any(term in message.lower() for term in ["severe", "massive", "major"]) else "Medium"
            },
            1: {
                "type": "Wildfire",
                "description": "Forest fires, bush fires, and other fire-related disasters",
                "severity": "High" if any(term in message.lower() for term in ["severe", "massive", "spreading rapidly"]) else "Medium"
            },
            2: {
                "type": "Earthquake",
                "description": "Seismic activities, tremors, and related structural damage",
                "severity": "High" if any(term in message.lower() for term in ["major", "massive", "strong"]) else "Medium"
            }
        }
        
        predicted_disaster = disaster_mapping.get(disaster_type, {"type": "Unknown", "description": "Unclassified disaster type"})

        # Find similar news with improved threshold
        similarity_threshold = 0.3  # Lowered threshold for better matching
        similar_news = []
        for idx, score in enumerate(similarity_scores[0]):
            if score > similarity_threshold:
                similar_news.append({
                    "message": sentences[idx]["message"],
                    "source": sentences[idx]["source"],
                    "similarity_score": float(score)
                })

        # Sort similar news by similarity score
        similar_news.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Get all keywords for method detection
        all_keywords = [term for terms in DISASTER_KEYWORDS.values() for term in terms]
        max_similarity = float(np.max(similarity_scores))

        return jsonify({
            "input_analysis": {
                "message": message,
                "source": source,
                "predicted_disaster": predicted_disaster,
                "confidence": {
                    "score": max_similarity,
                    "method": "not_disaster_related" if disaster_type == -1 else 
                             "keyword_match" if any(term in message.lower() for term in all_keywords) else 
                             "ml_classification"
                }
            },
            "similar_news": similar_news,
            "cluster_info": {
                "cluster_id": int(cluster[0]),
                "total_clusters": num_clusters
            },
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_used": "sentence-transformers/all-MiniLM-L6-v2",
                "confidence_score": max_similarity,
                "classification_threshold": 0.2
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_similarities', methods=['GET'])
def get_similarities():
    try:
        similarities = cosine_similarity(embeddings)
        return jsonify({
            "similarity_matrix": similarities.tolist(),
            "messages": [sentence["message"] for sentence in sentences]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)
