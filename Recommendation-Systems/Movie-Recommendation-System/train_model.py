import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Loading dataset...")
df = pd.read_csv("movies.csv")

# Remove noise
df = df[df["genres"] != "(no genres listed)"]

# First genre as label
df["genre"] = df["genres"].apply(lambda x: x.split("|")[0])

# Enriched text
df["text"] = df["title"] + " " + df["genres"].str.replace("|", " ", regex=False)

X = df["text"]
y = df["genre"]

print("Vectorizing with TF-IDF...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9
)

X_vec = vectorizer.fit_transform(X)

print("Training Naive Bayes...")
model = MultinomialNB()
model.fit(X_vec, y)

print("Saving model...")
pickle.dump(model, open("genre_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model retrained successfully")
