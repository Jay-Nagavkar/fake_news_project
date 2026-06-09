import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load Fake and Real news datasets
# 'on_bad_lines="skip"' skips problematic lines that might break the parser
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

# Drop rows where the 'text' field is missing
df = df.dropna(subset=["text"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=7)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Passive-Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save trained model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model training complete. Saved as 'model.pkl' and 'vectorizer.pkl'.")
