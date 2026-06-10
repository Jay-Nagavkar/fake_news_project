import requests
from flask import Flask, request, jsonify, render_template
import pickle
import os
from dotenv import load_dotenv
import numpy as np
from lime.lime_text import LimeTextExplainer

load_dotenv()

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Fetch API Key securely
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = f"http://api.mediastack.com/v1/news?access_key={NEWS_API_KEY}&countries=us&languages=en"

displayed_articles = set()

# Initialize LIME Explainer
explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

def predictor_pipeline(texts):
    """Pipeline required by LIME to get prediction probabilities"""
    vec = vectorizer.transform(texts)
    return model.predict_proba(vec)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.json.get("news_text", "").strip()

    if not news_text:
        return jsonify({"error": "No text provided!"}), 400

    try:
        text_vectorized = vectorizer.transform([news_text])

        if not hasattr(model, "predict_proba"):
            return jsonify({"error": "Model does not support confidence score."}), 500

        probabilities = model.predict_proba(text_vectorized)[0]
        prediction = model.predict(text_vectorized)[0]
        confidence = round(float(max(probabilities)) * 100, 2)

        result = "Real News 📰" if prediction == 1 else "Fake News 🚨"

        exp = explainer.explain_instance(news_text, predictor_pipeline, num_features=8)
        explanation = [
            {
                "word": word,
                "score": round(float(score), 4),
                "impact": "Real" if score > 0 else "Fake"
            }
            for word, score in exp.as_list()
        ]

        return jsonify({
            "prediction": result,
            "confidence": f"{confidence}%",
            "probabilities": {
                "fake": round(float(probabilities[0]) * 100, 2),
                "real": round(float(probabilities[1]) * 100, 2)
            },
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/live-news", methods=["GET"])
def live_news():
    """Fetches latest news from the API, with a safety fallback."""
    global displayed_articles
    try:
        # Added timeout=3. If the API doesn't respond in 3 seconds, it safely aborts instead of freezing your app.
        response = requests.get(NEWS_API_URL, timeout=3)
        news_data = response.json()

        if "data" not in news_data:
            raise ValueError("Invalid API Response")

        news_results = []
        for article in news_data["data"]:
            title = article.get("title", "")
            description = article.get("description", "")
            
            if title in displayed_articles:
                continue
            displayed_articles.add(title)

            combined_text = f"{title} {description}"
            
            text_vectorized = vectorizer.transform([combined_text])
            prediction = model.predict(text_vectorized)[0]
            result = "Real News 📰" if prediction == 1 else "Fake News 🚨"

            news_results.append({"title": title, "prediction": result})

            if len(news_results) >= 5:
                break

        return jsonify(news_results)

    except Exception as e:
        print(f"API unreachable, using fallback data. Reason: {e}")
        
        # FALLBACK DATA: If the API is blocked or internet is down, the UI still gets populated!
        fallback_news = [
            {"title": "Global Markets Rally as New Tech Innovations Surge", "prediction": "Real News 📰"},
            {"title": "BREAKING: Medical Insider Leaks Miracle Cure for All Illnesses!", "prediction": "Fake News 🚨"},
            {"title": "Federal Reserve Announces Stable Interest Rates for Q3", "prediction": "Real News 📰"},
            {"title": "SHOCKING: Scientists Clone Dinosaurs in Secret Underground Lab", "prediction": "Fake News 🚨"}
        ]
        return jsonify(fallback_news)

if __name__ == "__main__":
    app.run(debug=True)