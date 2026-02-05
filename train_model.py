import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- Load Data ----------------
df = pd.read_csv("spam.csv")

# Ensure correct column names
df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})

X = df['Message']
y = df['Category']

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Vectorization ----------------
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)

# ---------------- Train Model ----------------
model = MultinomialNB()
model.fit(X_train_count, y_train)

# ---------------- Save Files ----------------
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl and vectorizer.pkl created successfully")