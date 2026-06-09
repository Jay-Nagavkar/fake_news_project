import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets (skipping bad lines for robustness)
fake_df = pd.read_csv("Fake.csv", on_bad_lines="skip")
real_df = pd.read_csv("True.csv", on_bad_lines="skip")

# Add labels
fake_df["label"] = 0
real_df["label"] = 1

# Balance the dataset
min_len = min(len(fake_df), len(real_df))
fake_df = fake_df.sample(n=min_len, random_state=7)
real_df = real_df.sample(n=min_len, random_state=7)

# Merge datasets
df = pd.concat([fake_df, real_df], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)

# THE FIX: Combine Title and Text for a richer feature set
# (Assuming your dataset has a 'title' column, which the standard Fake/True datasets do)
df["full_text"] = df["title"] + " " + df["text"]
df = df.dropna(subset=["full_text"])

# Split data using the new combined feature
X_train, X_test, y_train, y_test = train_test_split(df["full_text"], df["label"], test_size=0.2, random_state=7)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Better Evaluation Metrics
y_pred = model.predict(X_test_tfidf)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save models
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model training complete. Saved as 'model.pkl' and 'vectorizer.pkl'.")