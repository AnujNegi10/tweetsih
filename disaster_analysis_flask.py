from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import numpy as np
from datetime import datetime
import spacy
# Initialize Flask app
app = Flask(__name__)
def extract_location_from_headline(headline):
    """
    Extracts locations from a given news headline using spaCy.

    :param headline: The news headline (string).
    :return: A list of detected locations.
    """
    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the headline
    doc = nlp(headline)
    
    # Extract locations
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    
    return locations

# Define global keywords dictionary
DISASTER_KEYWORDS = {
    "earthquake": ["earthquake", "tremor", "seismic", "magnitude", "richter"],
    "flood": ["flood", "flooding", "deluge", "submerged", "overflow"],
    "wildfire": ["fire", "wildfire", "bush fire", "forest fire", "flames"],
    "cyclone": ["cyclone", "hurricane", "typhoon", "storm", "tornado"],
    "landslide": ["landslide", "mudslide", "rockslide", "avalanche"],
    "tsunami": ["tsunami", "tidal wave", "ocean wave"],
    "lightning": ["lightning", "thunderstorm", "thunder", "electric storm"],
    "drought": ["drought", "water scarcity", "dry spell", "no water", "water shortage", "water crisis"],
    "heat wave": ["heat wave", "heatwave", "high temperature", "extreme heat"],
    "cold wave": ["cold wave", "cold spell", "low temperature", "freezing"]
}

# Additional context keywords to help disambiguation
CONTEXT_KEYWORDS = {
    "flood": ["heavy rain", "overflow", "rising water", "submerged"],
    "drought": ["shortage", "scarcity", "weeks without", "no access to water", "water crisis"]
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
    {"message": "Earthquake of magnitude 4.5 recorded.", "source": "Reuters"},    
    {"message": "Cyclone moving towards the eastern coast, residents advised to evacuate.", "source": "National Weather Service"},
    {"message": "Cyclone Yara causes severe damage to coastal villages.", "source": "BBC"},
    {"message": "Wind speeds of 150 km/h reported as cyclone makes landfall.", "source": "Reuters"},
    {"message": "Emergency shelters set up ahead of cyclone arrival.", "source": "Local News"},
    {"message": "Cyclone alert issued for southern coastal areas.", "source": "CNN"},
    {"message": "Landslide blocks main highway, leaving hundreds stranded.", "source": "AP News"},
    {"message": "Heavy rains trigger landslide in hilly region, damages reported.", "source": "Local News"},
    {"message": "Rescue operations underway after landslide buries village.", "source": "Reuters"},
    {"message": "Landslide warnings issued following continuous rainfall.", "source": "BBC"},
    {"message": "Roads remain closed due to landslide in mountain pass.", "source": "CNN"},
    {"message": "Tsunami waves expected after undersea earthquake, coastal areas on high alert.", "source": "USGS"},
    {"message": "Tsunami hits island nations, causing widespread destruction.", "source": "Reuters"},
    {"message": "Evacuation orders issued for residents in low-lying coastal areas.", "source": "National Weather Service"},
    {"message": "Tsunami warning canceled after seismic activity subsides.", "source": "AP News"},
    {"message": "Tsunami drills conducted in coastal towns to ensure preparedness.", "source": "BBC"},
    {"message": "Severe thunderstorm with frequent lightning reported in urban areas.", "source": "Weather Channel"},
    {"message": "Lightning strikes cause power outages in several districts.", "source": "Local News"},
    {"message": "Thunder and lightning disrupt flights at international airport.", "source": "Reuters"},
    {"message": "Precautions advised as lightning storms predicted tonight.", "source": "BBC"},
    {"message": "Farmers warned against outdoor activity during lightning storms.", "source": "AP News"},
    {"message": "Severe drought leaves thousands of acres of farmland barren.", "source": "Reuters"},
    {"message": "Water scarcity hits towns amid prolonged drought conditions.", "source": "CNN"},
    {"message": "Government announces relief package for drought-affected areas.", "source": "Local News"},
    {"message": "Drought worsens as reservoirs drop to critically low levels.", "source": "AP News"},
    {"message": "Farmers struggle with crop failures due to lack of rainfall.", "source": "BBC"},
    {"message": "Heat wave pushes temperatures to record highs in the region.", "source": "Weather Channel"},
    {"message": "Authorities advise staying indoors during peak heat wave hours.", "source": "Local News"},
    {"message": "Hospitals see a surge in heatstroke cases amid ongoing heat wave.", "source": "Reuters"},
    {"message": "Heat wave disrupts power supply as demand for cooling rises.", "source": "CNN"},
    {"message": "Schools closed for a week due to intense heat wave.", "source": "BBC"},
    {"message": "Cold wave grips northern region, temperatures drop below freezing.", "source": "Weather Channel"},
    {"message": "Homeless shelters filled to capacity as cold wave intensifies.", "source": "Local News"},
    {"message": "Cold wave causes frost damage to crops in agricultural zones.", "source": "AP News"},
    {"message": "Authorities distribute blankets to vulnerable groups during cold wave.", "source": "Reuters"},
    {"message": "Transportation disrupted due to heavy snow and freezing temperatures.", "source": "BBC"},


]

# Extract just messages for embedding
messages = [item["message"] for item in sentences]

# 3. Generate Sentence Embeddings
embeddings = model.encode(messages)

# 5. Disaster Type Classification
# Labels: 0 - Flood, 1 - Wildfire, 2 - Earthquake, 3 - Cyclone, 4 - Landslide, 
#         5 - Tsunami, 6 - Lightning, 7 - Drought, 8 - Heat Wave, 9 - Cold Wave
labels = [
    0, 1, 2, 0, 0,  # First 5 messages
    2, 2, 1, 0, 2,  # Next 5 messages
    3, 3, 3, 3, 3,  # Cyclone related messages
    4, 4, 4, 4, 4,  # Landslide related messages
    5, 5, 5, 5, 5,  # Tsunami related messages
    6, 6, 6, 6, 6,  # Lightning related messages
    7, 7, 7, 7, 7,  # Drought related messages
    8, 8, 8, 8, 8,  # Heat wave related messages
    9, 9, 9, 9, 9   # Cold wave related messages
]

# Train Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(embeddings, labels)

# 6. Clustering
num_clusters = 10  # Updated to match number of disaster types
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_results = kmeans.fit(embeddings)

# Optional: Dimensionality reduction for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Create a visualization of the clusters (commented out for production)
"""
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('Disaster Messages Clustering')
plt.colorbar(scatter)
plt.savefig('clusters.png')
plt.close()
"""

def predict_disaster_type(message, embedding, similarity_scores):
    message_lower = message.lower()
    
    # First check for drought-specific context
    for term in CONTEXT_KEYWORDS["drought"]:
        if term in message_lower:
            return 7  # Drought
            
    # Then check for flood-specific context
    for term in CONTEXT_KEYWORDS["flood"]:
        if term in message_lower:
            return 0  # Flood
    
    # Check other disaster keywords
    for disaster_type, terms in DISASTER_KEYWORDS.items():
        if any(term in message_lower for term in terms):
            if disaster_type == "earthquake":
                return 2
            elif disaster_type == "flood":
                return 0
            elif disaster_type == "wildfire":
                return 1
            elif disaster_type == "cyclone":
                return 3
            elif disaster_type == "landslide":
                return 4
            elif disaster_type == "tsunami":
                return 5
            elif disaster_type == "lightning":
                return 6
            elif disaster_type == "drought":
                return 7
            elif disaster_type == "heat wave":
                return 8
            elif disaster_type == "cold wave":
                return 9
    
    # If no clear keyword match, use ML prediction with similarity threshold
    max_similarity = float(np.max(similarity_scores))
    if max_similarity < 0.2:  # Threshold for non-disaster content
        return -1
        
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
            },
            3: {
                "type": "Cyclone",
                "description": "Cyclones, hurricanes, typhoons, and other storm-related disasters",
                "severity": "High" if any(term in message.lower() for term in ["severe", "massive", "major"]) else "Medium"
            },
            4: {
                "type": "Landslide",
                "description": "Landslides, mudslides, rockslides, and other geological disasters",
                "severity": "High" if any(term in message.lower() for term in ["severe", "massive", "major"]) else "Medium"
            },
            5: {
                "type": "Tsunami",
                "description": "Tsunamis and other ocean-related disasters",
                "severity": "High" if any(term in message.lower() for term in ["severe", "massive", "major"]) else "Medium"
            },
            6: {
                "type": "Lightning",
                "description": "Lightning storms and other electrical disasters",
                "severity": "High" if any(term in message.lower() for term in ["severe", "massive", "major"]) else "Medium"
            },
            7: {
                "type": "Drought",
                "description": "Droughts and other water scarcity-related disasters",
                "severity": "High" if any(term in message.lower() for term in ["severe", "massive", "major"]) else "Medium"
            },
            8: {
                "type": "Heat Wave",
                "description": "Heat waves and other temperature-related disasters",
                "severity": "High" if any(term in message.lower() for term in ["severe", "massive", "major"]) else "Medium"
            },
            9: {
                "type": "Cold Wave",
                "description": "Cold waves and other temperature-related disasters",
                "severity": "High" if any(term in message.lower() for term in ["severe", "massive", "major"]) else "Medium"
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

        location = extract_location_from_headline(message)

        return jsonify({
            "input_analysis": {
                "message": message,
                "location" :location,
                "source": source,
                "predicted_disaster": predicted_disaster,
                "confidence": {
                    "score": max_similarity,
                    "method": "not_disaster_related" if disaster_type == -1 else 
                             "keyword_match" if any(term in message.lower() for term in all_keywords) else 
                             "ml_classification"
                }
            },
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
    app.run(debug=True, port=3001)