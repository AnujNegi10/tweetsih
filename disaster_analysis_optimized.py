from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Define disaster keywords with expanded vocabulary
DISASTER_KEYWORDS = {
    "earthquake": [
        "earthquake", "tremor", "seismic", "magnitude", "richter", "quake", 
        "aftershock", "epicenter", "seismograph", "tectonic", "temblor"
    ],
    "flood": [
        "flood", "flooding", "deluge", "submerged", "overflow", "inundation",
        "torrential", "waterlogged", "flash flood", "rising water", "river overflow"
    ],
    "wildfire": [
        "fire", "wildfire", "bush fire", "forest fire", "flames", "blaze",
        "inferno", "burning", "combustion", "firestorm", "conflagration"
    ],
    "cyclone": [
        "cyclone", "hurricane", "typhoon", "storm", "tornado", "windstorm",
        "tropical storm", "twister", "gale", "whirlwind", "tempest"
    ],
    "landslide": [
        "landslide", "mudslide", "rockslide", "avalanche", "debris flow",
        "earth movement", "ground failure", "slope failure", "land slip"
    ],
    "tsunami": [
        "tsunami", "tidal wave", "ocean wave", "seismic sea wave",
        "harbor wave", "coastal flooding", "surge wave"
    ],
    "drought": [
        "drought", "water scarcity", "dry spell", "water shortage", "arid",
        "water crisis", "rainfall deficit", "water deficiency"
    ],
    "heat wave": [
        "heat wave", "heatwave", "extreme heat", "hot spell", "scorching",
        "sweltering", "thermal stress", "temperature surge"
    ],
    "cold wave": [
        "cold wave", "cold spell", "extreme cold", "freezing", "frost",
        "severe winter", "cold snap", "arctic blast"
    ]
}

# Expanded training dataset
disaster_data = [
    # Earthquakes
    {"message": "Massive earthquake of magnitude 7.2 strikes coastal region", "label": 0},
    {"message": "Aftershocks continue to rattle the area after main quake", "label": 0},
    {"message": "Buildings collapsed due to strong seismic activity", "label": 0},
    {"message": "Earthquake tremors felt across multiple cities", "label": 0},
    {"message": "Seismograph records significant tectonic movement", "label": 0},
    
    # Floods
    {"message": "Severe flooding has submerged entire neighborhoods", "label": 1},
    {"message": "Flash floods wash away bridges and roads", "label": 1},
    {"message": "Rising water levels force evacuation of low-lying areas", "label": 1},
    {"message": "Torrential rains cause widespread flooding", "label": 1},
    {"message": "River overflow leads to massive inundation", "label": 1},
    
    # Wildfires
    {"message": "Massive wildfire spreads through national forest", "label": 2},
    {"message": "Forest fire threatens residential areas", "label": 2},
    {"message": "Firefighters battle intense blaze in wilderness", "label": 2},
    {"message": "Bushfire creates firestorm in remote region", "label": 2},
    {"message": "Wildfire smoke affects air quality in nearby cities", "label": 2},
    
    # Cyclones
    {"message": "Hurricane approaches with 150 mph winds", "label": 3},
    {"message": "Typhoon causes extensive damage to coastal areas", "label": 3},
    {"message": "Tropical storm intensifies into major cyclone", "label": 3},
    {"message": "Tornado tears through suburban area", "label": 3},
    {"message": "Powerful windstorm uproots trees and power lines", "label": 3},
    
    # Landslides
    {"message": "Heavy rains trigger massive landslide", "label": 4},
    {"message": "Mudslide blocks major highway", "label": 4},
    {"message": "Rockslide damages mountain infrastructure", "label": 4},
    {"message": "Debris flow threatens village after rainfall", "label": 4},
    {"message": "Ground failure causes evacuation of hillside homes", "label": 4},
    
    # Tsunamis
    {"message": "Tsunami warning issued after undersea earthquake", "label": 5},
    {"message": "Massive ocean waves approach coastline", "label": 5},
    {"message": "Coastal areas evacuated due to tsunami threat", "label": 5},
    {"message": "Seismic sea wave detected in Pacific Ocean", "label": 5},
    {"message": "Harbor wave causes destruction in coastal towns", "label": 5},
    
    # Droughts
    {"message": "Severe drought affects agricultural production", "label": 6},
    {"message": "Water scarcity reaches critical levels", "label": 6},
    {"message": "Extended dry spell impacts reservoir levels", "label": 6},
    {"message": "Rainfall deficit leads to crop failure", "label": 6},
    {"message": "Water crisis worsens in rural areas", "label": 6},
    
    # Heat Waves
    {"message": "Extreme heat wave grips the region", "label": 7},
    {"message": "Temperature soars to record-breaking levels", "label": 7},
    {"message": "Scorching heat affects daily activities", "label": 7},
    {"message": "Thermal stress warning issued for elderly", "label": 7},
    {"message": "Sweltering conditions continue for fifth day", "label": 7},
    
    # Cold Waves
    {"message": "Severe cold wave brings life to standstill", "label": 8},
    {"message": "Arctic blast causes record low temperatures", "label": 8},
    {"message": "Extreme cold conditions affect transportation", "label": 8},
    {"message": "Frost damage reported in agricultural areas", "label": 8},
    {"message": "Cold snap leads to emergency shelter opening", "label": 8}
]

# Load Pre-Trained Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Prepare data
messages = [item["message"] for item in disaster_data]
labels = [item["label"] for item in disaster_data]

# Generate embeddings
embeddings = model.encode(messages)

# Split data
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Create ensemble model
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=42)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft'
)

# Train model
ensemble.fit(X_train, y_train)

# Evaluate model
y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

def predict_disaster_type(message, embedding, similarity_scores):
    message_lower = message.lower()
    severity_level = "Medium"
    
    # Check severity indicators
    high_severity_words = ["severe", "massive", "extreme", "critical", "emergency", 
                          "dangerous", "catastrophic", "devastating", "major"]
    
    if any(word in message_lower for word in high_severity_words):
        severity_level = "High"
    
    # Check keywords first
    for disaster_type, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in message_lower for keyword in keywords):
            disaster_mapping = {
                "earthquake": 0, "flood": 1, "wildfire": 2, "cyclone": 3,
                "landslide": 4, "tsunami": 5, "drought": 6, "heat wave": 7,
                "cold wave": 8
            }
            return disaster_mapping[disaster_type], severity_level
    
    # If no keywords match, use ensemble prediction
    max_similarity = float(np.max(similarity_scores))
    if max_similarity < 0.2:
        return -1, "None"
    
    prediction = ensemble.predict(embedding)[0]
    return prediction, severity_level

@app.route('/')
def home():
    return jsonify({
        "message": "Disaster Analysis API is running",
        "model_accuracy": f"{accuracy * 100:.2f}%",
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
        
        # Predict disaster type
        disaster_type, severity = predict_disaster_type(message, embedding, similarity_scores)
        
        # Map disaster type to description
        disaster_mapping = {
            -1: {
                "type": "Not Disaster Related",
                "description": "This message does not appear to be related to any natural disaster",
                "severity": "None"
            },
            0: {
                "type": "Earthquake",
                "description": "Seismic activity and related impacts",
                "severity": severity
            },
            1: {
                "type": "Flood",
                "description": "Water-related flooding and inundation",
                "severity": severity
            },
            2: {
                "type": "Wildfire",
                "description": "Forest fires and related incidents",
                "severity": severity
            },
            3: {
                "type": "Cyclone",
                "description": "Severe storms and cyclonic activity",
                "severity": severity
            },
            4: {
                "type": "Landslide",
                "description": "Ground movement and geological incidents",
                "severity": severity
            },
            5: {
                "type": "Tsunami",
                "description": "Seismic sea waves and coastal impacts",
                "severity": severity
            },
            6: {
                "type": "Drought",
                "description": "Water scarcity and related issues",
                "severity": severity
            },
            7: {
                "type": "Heat Wave",
                "description": "Extreme heat conditions",
                "severity": severity
            },
            8: {
                "type": "Cold Wave",
                "description": "Extreme cold conditions",
                "severity": severity
            }
        }
        
        predicted_disaster = disaster_mapping.get(disaster_type, {"type": "Unknown", "description": "Unclassified disaster type"})

        # Find similar messages
        similar_messages = []
        for idx, score in enumerate(similarity_scores[0]):
            if score > 0.3:
                similar_messages.append({
                    "message": messages[idx],
                    "similarity_score": float(score)
                })

        similar_messages.sort(key=lambda x: x["similarity_score"], reverse=True)

        return jsonify({
            "input_analysis": {
                "message": message,
                "source": source,
                "predicted_disaster": predicted_disaster,
                "model_confidence": float(np.max(similarity_scores)),
                "model_accuracy": f"{accuracy * 100:.2f}%"
            },
            "similar_messages": similar_messages[:5],
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": "Ensemble (RandomForest + GradientBoosting)",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
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
            "messages": messages,
            "model_accuracy": f"{accuracy * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Print initial model performance
    print("\nInitial Model Performance:")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nStarting Flask server...")
    app.run(debug=True, port=3000)
