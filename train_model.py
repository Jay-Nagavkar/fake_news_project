import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Fake and Real news datasets
fake_df = pd.read_csv("Fake.csv", on_bad_lines="skip")
real_df = pd.read_csv("True.csv", on_bad_lines="skip")

# Add a 'label' column (0 = Fake, 1 = Real)
fake_df["label"] = 0
real_df["label"] = 1

# Balance the dataset
min_len = min(len(fake_df), len(real_df))
fake_df = fake_df.sample(n=min_len, random_state=7)
real_df = real_df.sample(n=min_len, random_state=7)

# Merge datasets and shuffle
df = pd.concat([fake_df, real_df], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)

# Combine Title and Text for a richer feature set
df["full_text"] = df["title"] + " " + df["text"]
df = df.dropna(subset=["full_text"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["full_text"], df["label"], test_size=0.2, random_state=7)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# TRAIN LOGISTIC REGRESSION (Supports predict_proba for LIME)
print("Training Logistic Regression Model...")
model = LogisticRegression(max_iter=1000, random_state=7)
model.fit(X_train_tfidf, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save trained model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model training complete. Saved as 'model.pkl' and 'vectorizer.pkl'.")