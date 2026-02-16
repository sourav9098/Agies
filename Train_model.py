# train_model.py
import pandas as pd
import pickle
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("⏳ 1. Downloading Datasets from HuggingFace...")
# Load Deepset dataset (contains both safe (0) and injection (1) labels)
dataset = load_dataset("deepset/prompt-injections", split="train")
df = pd.DataFrame(dataset)

# Clean up dataframe
df = df[['text', 'label']]
df.dropna(inplace=True)

print(f"✅ Dataset loaded. Total rows: {len(df)}")
print(df['label'].value_counts()) # Show balance of 0s and 1s

# Split data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

print("\n⚙️ 2. Training the TF-IDF Vectorizer...")
# TF-IDF converts words to numbers. We look at unigrams and bigrams (1-2 words together)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("🧠 3. Training the Logistic Regression Model...")
classifier = LogisticRegression(class_weight='balanced')
classifier.fit(X_train_vec, y_train)

# Test Accuracy
y_pred = classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

print("\n💾 4. Saving Models for FastAPI...")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

print("✅ SUCCESS! 'vectorizer.pkl' and 'classifier.pkl' have been created.")