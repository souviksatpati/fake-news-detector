import streamlit as st
import pickle
import re
import os
import base64
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Set modern Streamlit page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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

# --- Article Fetching Logic (NO CHANGES TO SCRAPING LOGIC) ---
def fetch_article_text(url):
    """Fetches the visible text from a given URL using web scraping."""
    if not urlparse(url).scheme in ['http', 'https']:
        return None, "Invalid URL format. Must start with http:// or https://"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
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

# --- Prediction Function (NO CHANGES TO MODEL PREDICTION LOGIC) ---
def detect_fake_news(news_article):
    cleaned_text = preprocess_text(news_article)
    vectorized_text = loaded_vectorizer.transform([cleaned_text])
    prediction = loaded_model.predict(vectorized_text)[0]
    return 'REAL' if prediction == 1 else 'FAKE'

# --- 2. Custom Styling & Modern Design System ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_custom_styles():
    image_path = os.path.join(os.path.dirname(__file__), "images.jpeg")
    if os.path.exists(image_path):
        bin_str = get_base64_of_bin_file(image_path)
        bg_css = f"""
        [data-testid="stAppViewContainer"], .stApp {{
            background-image: linear-gradient(rgba(10, 14, 26, 0.75), rgba(10, 14, 26, 0.85)), url("data:image/jpeg;base64,{bin_str}") !important;
            background-size: cover !important;
            background-position: center center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}
        [data-testid="stHeader"] {{
            background: transparent !important;
        }}
        """
    else:
        bg_css = """
        [data-testid="stAppViewContainer"], .stApp {
            background: #0b0f19 !important;
        }
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
        
        {bg_css}

        /* Global Typography & Resets */
        html, body, [class*="css"], .stApp {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: #f1f5f9;
        }}

        /* Main Glassmorphism Content Card */
        .main .block-container {{
            max-width: 860px !important;
            padding: 2.2rem 2.2rem 2.5rem 2.2rem !important;
            background: rgba(15, 23, 42, 0.82) !important;
            backdrop-filter: blur(18px) !important;
            -webkit-backdrop-filter: blur(18px) !important;
            border-radius: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
            margin-top: 1.2rem !important;
            margin-bottom: 2rem !important;
        }}

        /* Hero Header Section */
        .hero-section {{
            text-align: center;
            padding-bottom: 1.2rem;
            margin-bottom: 1.2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .badge-pill {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            background: rgba(99, 102, 241, 0.14);
            border: 1px solid rgba(99, 102, 241, 0.35);
            color: #a5b4fc;
            padding: 5px 14px;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }}
        .badge-pulse {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #38bdf8;
            box-shadow: 0 0 8px #38bdf8;
            display: inline-block;
        }}
        .hero-title {{
            font-size: 2.25rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.025em !important;
            color: #ffffff !important;
            margin: 0 0 0.4rem 0 !important;
            line-height: 1.2 !important;
        }}
        .hero-title span {{
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hero-desc {{
            color: #94a3b8 !important;
            font-size: 0.95rem !important;
            max-width: 620px;
            margin: 0 auto !important;
            line-height: 1.55 !important;
        }}

        /* Streamlit Tabs Navigation */
        div[data-baseweb="tab-list"] {{
            background: rgba(30, 41, 59, 0.5) !important;
            padding: 5px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            gap: 6px !important;
            margin-bottom: 1.2rem !important;
        }}
        div[data-baseweb="tab-list"] button {{
            border-radius: 8px !important;
            color: #94a3b8 !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            padding: 8px 16px !important;
            transition: all 0.2s ease !important;
            border: none !important;
            background: transparent !important;
        }}
        div[data-baseweb="tab-list"] button[aria-selected="true"] {{
            background: rgba(99, 102, 241, 0.28) !important;
            color: #ffffff !important;
            border: 1px solid rgba(99, 102, 241, 0.45) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
        }}
        div[data-baseweb="tab-border"] {{
            display: none !important;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: transparent !important;
        }}

        /* Text Area & Text Input Styling */
        .stTextArea textarea, .stTextInput input {{
            background: rgba(15, 23, 42, 0.85) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            border-radius: 12px !important;
            padding: 14px 16px !important;
            font-size: 0.94rem !important;
            line-height: 1.6 !important;
            font-family: 'Inter', sans-serif !important;
            transition: all 0.2s ease !important;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        }}
        .stTextArea textarea:focus, .stTextInput input:focus {{
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25), inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
            outline: none !important;
        }}
        .stTextArea textarea::placeholder, .stTextInput input::placeholder {{
            color: #64748b !important;
            font-style: normal !important;
        }}
        .stTextArea label, .stTextInput label {{
            color: #e2e8f0 !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            margin-bottom: 6px !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }}

        /* Custom Action Buttons */
        div.stButton > button {{
            background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 50%, #06b6d4 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            font-weight: 700 !important;
            font-size: 0.96rem !important;
            letter-spacing: 0.02em !important;
            box-shadow: 0 8px 24px -4px rgba(79, 70, 229, 0.5) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
            width: 100% !important;
            margin-top: 0.8rem !important;
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 28px -4px rgba(79, 70, 229, 0.7), 0 0 16px rgba(6, 182, 212, 0.45) !important;
            border-color: rgba(255, 255, 255, 0.4) !important;
        }}
        div.stButton > button:active {{
            transform: translateY(1px) scale(0.99) !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
        }}

        /* Quick Sample Chips */
        .sample-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 0.6rem;
            flex-wrap: wrap;
        }}
        .sample-label {{
            font-size: 0.82rem;
            color: #94a3b8;
            font-weight: 600;
        }}

        /* Modern Result Cards */
        .result-container {{
            margin-top: 1.5rem;
            border-radius: 16px;
            padding: 1.6rem;
            animation: slideIn 0.35s ease-out;
        }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .real-card {{
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.45) 0%, rgba(4, 47, 46, 0.55) 100%);
            border: 1px solid rgba(16, 185, 129, 0.45);
            box-shadow: 0 12px 30px rgba(6, 78, 59, 0.3);
        }}
        .fake-card {{
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.45) 0%, rgba(69, 10, 10, 0.55) 100%);
            border: 1px solid rgba(239, 68, 68, 0.45);
            box-shadow: 0 12px 30px rgba(127, 29, 29, 0.3);
        }}
        .result-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.9rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        .result-badge {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 5px 14px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        .real-badge {{
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.5);
            color: #34d399;
        }}
        .fake-badge {{
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.5);
            color: #f87171;
        }}
        .tag-capsule {{
            font-size: 0.78rem;
            color: #94a3b8;
            background: rgba(255, 255, 255, 0.06);
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .result-heading {{
            font-size: 1.45rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.4rem;
        }}
        .result-paragraph {{
            font-size: 0.92rem;
            color: #cbd5e1;
            line-height: 1.55;
            margin-bottom: 1.2rem;
        }}
        .result-metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.75rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .metric-cell {{
            background: rgba(0, 0, 0, 0.3);
            padding: 0.65rem 0.9rem;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.06);
        }}
        .metric-cell-title {{
            display: block;
            font-size: 0.74rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 2px;
        }}
        .metric-cell-val {{
            font-size: 0.96rem;
            font-weight: 700;
            color: #f8fafc;
        }}
        .val-real {{ color: #34d399; }}
        .val-fake {{ color: #f87171; }}

        /* System Info Cards */
        .info-card {{
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
        }}
        .info-card h4 {{
            color: #38bdf8;
            font-size: 1rem;
            margin: 0 0 0.5rem 0;
            font-weight: 700;
        }}
        .info-card p, .info-card li {{
            color: #cbd5e1;
            font-size: 0.88rem;
            line-height: 1.55;
            margin: 0.2rem 0;
        }}

        /* Modernized Alert Boxes */
        .stAlert {{
            border-radius: 12px !important;
            background: rgba(15, 23, 42, 0.9) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }}

        /* Neutral Project Footer */
        .app-footer {{
            margin-top: 2rem;
            padding-top: 1.2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            text-align: center;
        }}
        .footer-text {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
            color: #94a3b8;
            font-size: 0.82rem;
        }}
        .footer-sep {{
            color: #475569;
        }}
        .footer-badge {{
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #a5b4fc;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.76rem;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- 3. Streamlit Application Interface ---
set_custom_styles()

# Header Section
st.markdown("""
<div class="hero-section">
    <div class="badge-pill">
        <span class="badge-pulse"></span>
        AI-POWERED VERIFICATION SYSTEM
    </div>
    <h1 class="hero-title">Fake News <span>Detection System</span></h1>
    <p class="hero-desc">
        Analyze news headlines and full articles in real time using Natural Language Processing and Machine Learning to detect misinformation and verify source authenticity.
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_text, tab_url, tab_info = st.tabs([
    "📝 Direct Text Input",
    "🔗 Analyze by URL",
    "📊 Model & Pipeline Info"
])

# Quick Sample Handlers
SAMPLE_REAL = (
    "WASHINGTON (Reuters) - The U.S. Senate on Thursday passed a comprehensive bipartisan bill "
    "allocating investments into nationwide clean energy infrastructure and modern electric grid resilience, "
    "sending the legislation to the president to be signed into federal law. Leaders from both committees "
    "commended the agreement after months of cross-party negotiations."
)

SAMPLE_FAKE = (
    "Shocking top secret documents leaked from an underground research bunker reveal that mysterious alien beings "
    "have infiltrated world leadership councils and are secretly manipulating global financial markets and weather "
    "patterns using high frequency microwave transmitters disguised as streetlights."
)

# Tab 1: Direct Text Input
with tab_text:
    # Quick sample loader buttons
    col_s1, col_s2, col_s3 = st.columns([1, 1, 0.7])
    with col_s1:
        if st.button("📰 Load Real Sample", key="btn_sample_real"):
            st.session_state['news_text_val'] = SAMPLE_REAL
    with col_s2:
        if st.button("⚠️ Load Fake Sample", key="btn_sample_fake"):
            st.session_state['news_text_val'] = SAMPLE_FAKE
    with col_s3:
        if st.button("🧹 Clear", key="btn_sample_clear"):
            st.session_state['news_text_val'] = ""

    default_text = st.session_state.get('news_text_val', '')

    text_input_content = st.text_area(
        "News Article Text or Headline:",
        value=default_text,
        height=190,
        placeholder="Paste news headline or full article text here to evaluate authenticity (e.g., Officials announce new scientific breakthroughs in clean energy storage during the international summit...)",
        key="direct_text_area"
    )

    btn_analyze_text = st.button("🔍 Verify Article Text", key="btn_analyze_direct")

# Tab 2: URL Input
with tab_url:
    url_input_content = st.text_input(
        "News Article URL:",
        placeholder="e.g., https://www.reuters.com/world/news-article-example",
        key="news_url_field"
    )
    st.markdown("""
    <p style='color: #94a3b8; font-size: 0.82rem; margin-top: -0.4rem;'>
        💡 The system will automatically fetch the article text from the web page and run it through the classification pipeline.
    </p>
    """, unsafe_allow_html=True)
    
    btn_analyze_url = st.button("🌐 Scrape & Verify URL", key="btn_analyze_url")

# Tab 3: Model & Pipeline Info
with tab_info:
    st.markdown("""
    <div class="info-card">
        <h4>🤖 Machine Learning Model Architecture</h4>
        <p><strong>Algorithm:</strong> Passive Aggressive Classifier (PAC)</p>
        <p><strong>Evaluation Accuracy:</strong> 99.47% on Kaggle Fake & Real News Dataset</p>
        <p>Passive Aggressive algorithms are online learning techniques well-suited for high-dimensional text streams. The model remains passive upon correct classification and aggressively updates its weights when encountering a prediction error.</p>
    </div>
    <div class="info-card">
        <h4>⚙️ NLP Preprocessing Pipeline</h4>
        <ul>
            <li><strong>Text Normalization:</strong> Lowercasing and non-alphabetic symbol removal via Regex.</li>
            <li><strong>Stopword Filtering:</strong> Stripping common English non-informative words using NLTK English stopwords.</li>
            <li><strong>Porter Stemming:</strong> Reducing derived words to their grammatical root forms.</li>
            <li><strong>Feature Engineering:</strong> TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Prediction Execution Logic
article_to_analyze = None
source_label = None

if btn_analyze_text:
    if text_input_content and len(text_input_content.strip()) > 10:
        article_to_analyze = text_input_content
        source_label = "Direct Text Input"
    else:
        st.warning("⚠️ Please provide news headline or article text (at least 10 characters) to analyze.")

elif btn_analyze_url:
    if url_input_content:
        with st.spinner("Fetching and extracting article content from URL..."):
            scraped_text, error_msg = fetch_article_text(url_input_content)
            
        if error_msg:
            st.error(f"Scraping Failed: {error_msg}")
        elif scraped_text:
            article_to_analyze = scraped_text
            source_label = "Web Article Scraper"
    else:
        st.warning("⚠️ Please paste a valid article URL starting with http:// or https://")

# Run Classification & Render Results
if article_to_analyze:
    with st.spinner("Processing natural language features and running PAC model..."):
        prediction_result = detect_fake_news(article_to_analyze)

    # Word and character count statistics
    word_count = len(article_to_analyze.split())
    char_count = len(article_to_analyze)

    if prediction_result == 'REAL':
        st.markdown(f"""
        <div class="result-container real-card">
            <div class="result-header">
                <div class="result-badge real-badge">
                    <span>✓</span> VERIFIED AUTHENTIC
                </div>
                <div class="tag-capsule">Source: {source_label}</div>
            </div>
            <div class="result-heading">Classification: Authentic News (REAL)</div>
            <p class="result-paragraph">
                The vocabulary distribution, term frequencies, and linguistic structure match authentic, verified reporting standards observed in the training corpus.
            </p>
            <div class="result-metrics-grid">
                <div class="metric-cell">
                    <span class="metric-cell-title">Prediction Result</span>
                    <span class="metric-cell-val val-real">✅ Real News</span>
                </div>
                <div class="metric-cell">
                    <span class="metric-cell-title">Model Architecture</span>
                    <span class="metric-cell-val">Passive Aggressive</span>
                </div>
                <div class="metric-cell">
                    <span class="metric-cell-title">Training Accuracy</span>
                    <span class="metric-cell-val">99.47%</span>
                </div>
                <div class="metric-cell">
                    <span class="metric-cell-title">Content Volume</span>
                    <span class="metric-cell-val">{word_count} words ({char_count} chars)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-container fake-card">
            <div class="result-header">
                <div class="result-badge fake-badge">
                    <span>⚠</span> POTENTIAL MISINFORMATION
                </div>
                <div class="tag-capsule">Source: {source_label}</div>
            </div>
            <div class="result-heading">Classification: Fake News Flagged (FAKE)</div>
            <p class="result-paragraph">
                The linguistic phrasing, vocabulary patterns, and semantic density exhibit strong characteristics commonly associated with unreliable or fabricated news articles.
            </p>
            <div class="result-metrics-grid">
                <div class="metric-cell">
                    <span class="metric-cell-title">Prediction Result</span>
                    <span class="metric-cell-val val-fake">❌ Fake News</span>
                </div>
                <div class="metric-cell">
                    <span class="metric-cell-title">Model Architecture</span>
                    <span class="metric-cell-val">Passive Aggressive</span>
                </div>
                <div class="metric-cell">
                    <span class="metric-cell-title">Training Accuracy</span>
                    <span class="metric-cell-val">99.47%</span>
                </div>
                <div class="metric-cell">
                    <span class="metric-cell-title">Content Volume</span>
                    <span class="metric-cell-val">{word_count} words ({char_count} chars)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align: center; font-size: 0.82rem; color: #94a3b8; margin-top: 1rem; font-style: italic;'>
        Note: Machine learning models provide statistical predictions based on training data. Always verify critical news with recognized primary sources.
    </p>
    """, unsafe_allow_html=True)

# Footer (Neutral College/Academic Project Branding)
st.markdown("""
<footer class="app-footer">
    <div class="footer-text">
        <span>Fake News Detection System</span>
        <span class="footer-sep">•</span>
        <span>Natural Language Processing & Machine Learning</span>
        <span class="footer-sep">•</span>
        <span class="footer-badge">PAC Model: 99.47% Accuracy</span>
    </div>
</footer>
""", unsafe_allow_html=True)