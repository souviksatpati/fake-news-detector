import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# --- NEW IMPORTS FOR URL SCRAPING ---
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# --- 1. Load Resources and Set up Preprocessing (NO CHANGES TO MODEL LOGIC) ---
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

# --- NEW FUNCTION: Article Fetching Logic ---
def fetch_article_text(url):
    """Fetches the visible text from a given URL using web scraping."""
    # Basic URL validation
    if not urlparse(url).scheme in ['http', 'https']:
        return None, "Invalid URL format. Must start with http:// or https://"
    
    try:
        # 1. Fetch HTML content using a User-Agent header to mimic a browser
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes (4xx, 5xx)

        # 2. Parse HTML and extract text
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Simple extraction: get all visible text
        article_text = soup.get_text(separator=' ', strip=True)
        
        if len(article_text) < 100:
             return None, "The scraping retrieved too little text (less than 100 characters). The website might be JavaScript-heavy or blocked."
             
        return article_text, None
        
    except requests.exceptions.Timeout:
        return None, "Request timed out. The URL took too long to respond."
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP Error: Could not access the URL. Status Code: {e.response.status_code}."
    except requests.exceptions.RequestException as e:
        return None, f"Error accessing the URL: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred during scraping: {e}"


# --- 2. Prediction Function (NO CHANGES HERE) ---
def detect_fake_news(news_article):
    cleaned_text = preprocess_text(news_article)
    vectorized_text = loaded_vectorizer.transform([cleaned_text])
    prediction = loaded_model.predict(vectorized_text)[0]
    return 'REAL' if prediction == 1 else 'FAKE'


# --- 3. Custom Styling Function (YOUR EXISTING CSS) ---
def set_custom_styles():
    st.markdown(
        """
        <style>
        /* ... (Your full CSS block from the previous step) ... */
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
        
        /* Styling for Text Area and URL Input */
        .stTextArea label, .stTextInput label {
            color: #fff !important;
            font-family: 'Times New Roman', Times, serif; /* Changed to Times New Roman */
        }
        .stTextArea textarea, .stTextInput input {
            font-family: 'Times New Roman', Times, serif; /* Apply Times New Roman to content */
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

# --- 4. Streamlit UI Definition (Updated to include URL input) ---
set_custom_styles() 

st.title("📰 Fake News Detector")
st.markdown("---")

st.markdown("""
<p style='text-align: center; font-size: 1.1em; color: #f0f0f0; font-family: "Times New Roman", Times, serif;'>
    Analyze a news source by pasting the URL or pasting the text directly.
</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([0.5, 6, 0.5]) 

with col2:
    # NEW: URL Input Field
    news_url = st.text_input("Paste the News Article URL Here:", 
                             placeholder="e.g., https://www.nytimes.com/article...",
                             key="news_url")
    
    st.markdown("<p style='text-align: center; color: #aaa;'>— OR —</p>", unsafe_allow_html=True)

    # Existing: Text Area for User Input
    news_input = st.text_area("Paste the News Article Text Here:", height=200, 
                              placeholder='"A new study released today proves that eating chocolate is the secret to eternal life..."',
                              key="news_input")
    
    # Prediction Button
    analyze_button = st.button('Analyze News')

    st.markdown("---")

    # Display Analysis Result Logic (Updated to handle both inputs)
    if analyze_button:
        article_to_analyze = None
        error_message = None

        if news_url:
            # 1. Scrape the article from the URL
            with st.spinner('Fetching and scraping article from URL...'):
                article_to_analyze, error_message = fetch_article_text(news_url)
                
            if error_message:
                st.error(f"Scraping Failed: {error_message}")
                # Do not proceed with prediction
                st.stop()
            elif article_to_analyze:
                st.info("Article content successfully retrieved from URL.")

        elif news_input:
            # 2. Use direct text input
            article_to_analyze = news_input
            
        else:
            st.warning("Please provide a URL or paste text into the box to analyze.")
            st.stop()
            
        # 3. Run the prediction on the retrieved/pasted text
        if article_to_analyze and len(article_to_analyze.strip()) > 10:
            with st.spinner('Running prediction model...'):
                result = detect_fake_news(article_to_analyze)

            st.markdown("### Analysis Result:")

            if result == 'REAL':
                st.success(f"Classification: **✅ {result}**")
            else:
                st.error(f"Classification: **❌ {result}**")

            st.markdown(
                "<p style='text-align: center; font-size: 0.9em; color: #bbb; font-style: italic; font-family: \"Times New Roman\", Times, serif;'>Model can make mistakes. Always verify the results.</p>",
                unsafe_allow_html=True
            )
        elif article_to_analyze:
            # This handles the case where scraping retrieved too little text (error already handled)
            st.warning("The content found was too short to analyze.")


# Footer
st.markdown("<p style='text-align: center; font-size: 0.8em; color: #bbb; font-style: italic; font-family: \"Times New Roman\", Times, serif;'>Developed by Dee</p>", unsafe_allow_html=True)