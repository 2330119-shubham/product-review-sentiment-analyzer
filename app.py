import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup NLTK
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words])

# Load Model Artifacts
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("🛒 Product Review Sentiment Analyzer")
st.write("Enter a customer review below to analyze its sentiment.")

user_review = st.text_area("Customer Review:", "")

if st.button("Analyze Sentiment"):
    if user_review.strip() != "":
        cleaned = clean_text(user_review)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        
        if prediction == 'Positive':
            st.success(f"Result: 😊 **{prediction} Sentiment**")
        elif prediction == 'Negative':
            st.error(f"Result: 😡 **{prediction} Sentiment**")
        else:
            st.warning(f"Result: 😐 **{prediction} Sentiment**")
    else:
        st.info("Please enter some text to analyze.")
