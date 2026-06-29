import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("genre_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Movie Genre Predictor 🎬", page_icon="🎬")

st.title("🎬 Movie Genre Prediction")
st.write("Predict movie genre using **Naive Bayes**")

movie_title = st.text_input("Enter Movie Title:")

if st.button("Predict Genre"):
    if movie_title.strip() == "":
        st.warning("Please enter a movie title")
    else:
        data = vectorizer.transform([movie_title])
        prediction = model.predict(data)
        st.success(f"🎥 Predicted Genre: **{prediction[0]}**")
