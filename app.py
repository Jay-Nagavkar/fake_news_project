import requests
from flask import Flask, request, jsonify, render_template
import pickle

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the key securely
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = f"http://api.mediastack.com/v1/news?access_key={NEWS_API_KEY}&countries=us&languages=en"

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Updated News API Key (Replace with your actual MediaStack API key)
NEWS_API_KEY = "e8f41bb4d0f04953c2da5fdf562ea9fb"
NEWS_API_URL = f"http://api.mediastack.com/v1/news?access_key={NEWS_API_KEY}&countries=us&languages=en"

# Keep track of displayed articles
displayed_articles = set()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.json.get("news_text", "")

    if not news_text:
        return jsonify({"error": "No text provided!"})

    text_vectorized = vectorizer.transform([news_text])
    prediction = model.predict(text_vectorized)[0]
    result = "Real News 📰" if prediction == 1 else "Fake News 🚨"

    return jsonify({"prediction": result})

@app.route("/live-news", methods=["GET"])
def live_news():
    """Fetches latest news from the API and classifies them."""
    global displayed_articles
    try:
        response = requests.get(NEWS_API_URL)
        news_data = response.json()

        if "data" not in news_data:
            return jsonify({"error": "Could not fetch news."})

        news_results = []
        for article in news_data["data"]:
            title = article["title"]

            # Avoid showing duplicate news
            if title in displayed_articles:
                continue
            displayed_articles.add(title)

            text_vectorized = vectorizer.transform([title])
            prediction = model.predict(text_vectorized)[0]
            result = "Real News 📰" if prediction == 1 else "Fake News 🚨"

            news_results.append({"title": title, "prediction": result})

            # Limit to 5 news items
            if len(news_results) >= 5:
                break

        return jsonify(news_results)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
