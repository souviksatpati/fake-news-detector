import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- 1. Load Resources and Set up Preprocessing ---
# Define NLTK resources and setup preprocessing environment

# We use nltk.download() directly. If the resource is already present, NLTK
# will quickly confirm it is 'up-to-date!' and continue.
try:
    # Ensure 'stopwords' is available
    nltk.download('stopwords', quiet=True) 
    # Ensure 'punkt' is available, which is sometimes needed for general NLP tasks
    nltk.download('punkt', quiet=True)

except Exception as e:
    # Catch any generic error during download
    st.error(f"Error downloading NLTK resources: {e}")
    st.stop()


STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

# IMPORTANT: This function MUST be identical to the one used for training
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [ps.stem(word) for word in tokens if word not in STOPWORDS]
    return ' '.join(cleaned_tokens)


# Load the trained TF-IDF Vectorizer and Model
@st.cache_resource
def load_resources():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
            loaded_vectorizer = pickle.load(vectorizer_file)

        with open('fake_news_detector_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        
        return loaded_vectorizer, loaded_model

    except FileNotFoundError:
        st.error("Error: Model or Vectorizer file not found. Ensure .pkl files are present.")
        st.stop()

loaded_vectorizer, loaded_model = load_resources()


# --- 2. Prediction Function ---
def detect_fake_news(news_article):
    # 1. Clean the input text
    cleaned_text = preprocess_text(news_article)

    # 2. Vectorize the cleaned text
    vectorized_text = loaded_vectorizer.transform([cleaned_text])

    # 3. Predict the label (0 or 1)
    prediction = loaded_model.predict(vectorized_text)[0]
    
    return 'REAL' if prediction == 1 else 'FAKE'


# --- 3. Streamlit UI Definition ---
st.title("📰 Fake News Detector")
st.markdown("Enter a news article or snippet below to determine if it is likely Real or Fake.")
st.markdown("---")

# Text Area for User Input
news_input = st.text_area("Paste the News Article Here:", height=250, 
                          placeholder="Example: 'A new study released today proves that eating chocolate is the secret to eternal life.'")

# Prediction Button
if st.button('Analyze News'):
    if news_input:
        # Get the prediction
        result = detect_fake_news(news_input)

        st.markdown("### Analysis Result:")

        if result == 'REAL':
            st.success(f"Classification: **{result}**")
            st.balloons()
        else:
            st.error(f"Classification: **{result}**")
            st.snow()

    else:
        st.warning("Please paste some text into the box to analyze.")