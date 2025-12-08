import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- 1. Load Resources and Set up Preprocessing ---
try:
    nltk.download('stopwords', quiet=True) 
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")
    st.stop()

STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

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
    cleaned_text = preprocess_text(news_article)
    vectorized_text = loaded_vectorizer.transform([cleaned_text])
    prediction = loaded_model.predict(vectorized_text)[0]
    return 'REAL' if prediction == 1 else 'FAKE'


# --- 3. Custom Styling Function ---
def set_custom_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');
        
        /* Global Background and Fonts */
        .stApp {
            background: #000000; /* Proper dark black background */
            font-family: 'Times New Roman', Times, serif; /* Entire website font to Times New Roman */
            color: #fff;
        }
        
        /* Glassmorphism Effect for Main Container/Card */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: rgba(255, 255, 255, 0.1); /* Semi-transparent white */
            backdrop-filter: blur(10px); /* Glassmorphism blur */
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        }

        /* Styling for Headers */
        h1, h3 {
            color: #FFD700; /* Gold color */
            text-align: center;
            font-family: 'Times New Roman', Times, serif; /* Changed to Times New Roman */
        }
        
        /* Custom Button Styling with Animation and Glassmorphism */
        div.stButton > button {
            background: rgba(255, 255, 255, 0.15); /* Glassmorphism background */
            backdrop-filter: blur(5px); /* Glassmorphism blur */
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 10px 24px;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); /* Softer shadow */
            display: block; /* Ensure button takes full width of its container */
            margin-left: auto; /* Center the button */
            margin-right: auto; /* Center the button */
            font-family: 'Times New Roman', Times, serif; /* Changed to Times New Roman */
        }
        
        /* Button Hover and Active Effects */
        div.stButton > button:hover {
            background: rgba(255, 255, 255, 0.25);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
            transform: translateY(-2px);
        }
        div.stButton > button:active {
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            transform: translateY(1px); /* Button press animation */
        }
        
        /* Styling for Text Area */
        .stTextArea label {
            color: #fff !important;
            font-family: 'Times New Roman', Times, serif; /* Changed to Times New Roman */
        }
        .stTextArea textarea {
            font-family: 'Times New Roman', Times, serif; /* Apply Times New Roman to text area content */
        }

        /* Styling for Success/Error Messages */
        .stSuccess, .stError {
            border-radius: 10px;
            padding: 15px;
            font-size: 1.1em;
            font-weight: 600;
            font-family: 'Times New Roman', Times, serif; /* Changed to Times New Roman */
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# --- 4. Streamlit UI Definition ---
set_custom_styles() # Inject the custom styles first!

st.title("📰 Fake News Detector")
st.markdown("---")

st.markdown("""
<p style='text-align: center; font-size: 1.1em; color: #f0f0f0; font-family: "Times New Roman", Times, serif;'>
    Paste a news article or snippet below. Our model will analyze the language 
    patterns and determine if it aligns with "Real" or "Fake" news characteristics.
</p>
""", unsafe_allow_html=True)

# Use columns for centering the input area better and increasing width
col1, col2, col3 = st.columns([0.5, 6, 0.5]) # Increased middle column width

with col2:
    # Text Area for User Input
    news_input = st.text_area("Paste the News Article Here:", height=250, 
                              placeholder="Example: 'A new study released today proves that eating chocolate is the secret to eternal life.'",
                              key="news_input")
    
    # Prediction Button is placed centrally using CSS, removing redundant markdown wrapper
    analyze_button = st.button('Analyze News')

    st.markdown("---")

    # Display Analysis Result
    if analyze_button:
        if news_input:
            with st.spinner('Analyzing patterns...'):
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

# Footer
st.markdown("<p style='text-align: center; font-size: 0.8em; color: #bbb; font-style: italic; font-family: \"Times New Roman\", Times, serif;'>Developed by Dee</p>", unsafe_allow_html=True)