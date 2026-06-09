# 🚨 Real-Time Fake News Detector (XAI Powered)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-green)
![Machine Learning](https://img.shields.io/badge/Scikit--Learn-Model-orange)
![LIME](https://img.shields.io/badge/LIME-Explainable_AI-purple)

## 📌 Overview
This project is an end-to-end Machine Learning web application designed to classify news articles as **Real** or **Fake**. Moving beyond black-box classification, this application integrates **Explainable AI (XAI)** using LIME (Local Interpretable Model-agnostic Explanations) to provide visual transparency into the model's decision-making process.

It features two core functionalities:
1. **Manual Text Analysis:** Users can paste any article text to receive a prediction, a confidence score, and a breakdown of which specific words influenced the AI's decision.
2. **Live News Feed:** Integrates with the MediaStack API to pull real-time global news and classify headlines on the fly.

## ✨ Key Features
* **Explainable AI (LIME):** Highlights the exact words driving the classification (Green for "Real" indicators, Red for "Fake" indicators).
* **High-Accuracy ML Pipeline:** Utilizes a TF-IDF vectorizer paired with a Logistic Regression classifier, optimized for probability distribution and F1-score balance.
* **Real-Time Data Ingestion:** Asynchronous JavaScript fetching classifies live API news data every 15 seconds.
* **Responsive UI:** Clean, modern, vanilla CSS/JS frontend interface.

## 🧠 System Architecture
* **Data Preprocessing:** ISOT Fake News Dataset (Balanced), text cleaning, and TF-IDF Vectorization (max_df=0.7, English stop words removed).
* **Model:** `LogisticRegression(max_iter=1000)` chosen specifically over standard margin classifiers to output accurate `predict_proba` metrics required by LIME.
* **Backend:** Flask API handling model inference and external API requests.
* **Frontend:** DOM manipulation to render XAI tags and dynamic confidence scores.

## 🚀 Local Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/fake-news-detector.git](https://github.com/yourusername/fake-news-detector.git)
cd fake-news-detector

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
Create a .env file in the root directory and add your MediaStack API key:
NEWS_API_KEY=your_api_key_here

python app.py

Built by Jay — Artificial Intelligence and Data Science Engineering.
