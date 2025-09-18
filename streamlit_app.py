import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

# Streamlit page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()

# Load model components
@st.cache_resource
def load_model():
    try:
        model_components = joblib.load('sentiment_model_ridge (1).joblib')
        return model_components, True
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'sentiment_model_ridge.joblib' exists.")
        return None, False
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, False

# Copy your SentimentPreprocessor class here (sama seperti di Flask app)
class SentimentPreprocessor:
    def __init__(self):
        """Initialize preprocessor with same parameters as notebook"""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.regex_pattern = r'[^\w\s]|[\d]'
    
    # Copy semua method dari Flask app Anda
    def preprocess_text(self, text, text_type='text'):
        # ... (copy exact code from your Flask app)
        pass
    
    def predict_sentiment(self, text_content, platform='Twitter', location='USA', 
                         language='English', day_of_week='Monday', 
                         mentions='', hashtags='', brand_name='Unknown', 
                         product_name='Unknown', campaign_name='Unknown', 
                         campaign_phase='Launch', likes_count=0, shares_count=0, 
                         comments_count=0, impressions=1000, engagement_rate=0.1,
                         toxicity_score=0.1, user_past_sentiment_avg=0.5,
                         user_engagement_growth=0.1, buzz_change_rate=5.0):
        # ... (copy exact code from your Flask app)
        pass

# Initialize
model_components, model_available = load_model()
if model_available:
    preprocessor = SentimentPreprocessor()
    # Extract model components
    ridge_model = model_components['ridge_model']
    scaler = model_components['scaler']
    tfidf_text_vectorizer = model_components['tfidf_text_vectorizer']
    # ... extract other components

# Main UI
st.title("🎭 Sentiment Analysis App")
st.markdown("Analyze sentiment from text using machine learning")

if not model_available:
    st.error("Model not available. Please check your model file.")
    st.stop()

# Sidebar for model info
with st.sidebar:
    st.header("📊 Model Information")
    st.info(f"**Model Type:** {model_components.get('model_type', 'Ridge Regression')}")
    st.info(f"**Best Alpha:** {model_components['model_performance']['best_alpha']}")
    st.info(f"**R² Score:** {model_components['model_performance']['r2_score']:.4f}")

# Main form
with st.form("sentiment_form"):
    st.subheader("Enter Text for Analysis")
    
    # Text input
    text_content = st.text_area(
        "Text Content",
        placeholder="Enter the text you want to analyze...",
        height=100
    )
    
    # Two columns for other inputs
    col1, col2 = st.columns(2)
    
    with col1:
        platform = st.selectbox("Platform", ["Twitter", "Facebook", "Instagram", "LinkedIn"])
        location = st.selectbox("Location", ["USA", "UK", "Canada", "Australia"])
        language = st.selectbox("Language", ["English", "Spanish", "French"])
        
    with col2:
        day_of_week = st.selectbox("Day of Week", 
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        mentions = st.text_input("Mentions", placeholder="@username")
        hashtags = st.text_input("Hashtags", placeholder="#hashtag")
    
    # Advanced options (collapsible)
    with st.expander("Advanced Options"):
        col3, col4 = st.columns(2)
        
        with col3:
            brand_name = st.text_input("Brand Name", value="Unknown")
            product_name = st.text_input("Product Name", value="Unknown")
            campaign_name = st.text_input("Campaign Name", value="Unknown")
            
        with col4:
            likes_count = st.number_input("Likes Count", min_value=0, value=0)
            shares_count = st.number_input("Shares Count", min_value=0, value=0)
            comments_count = st.number_input("Comments Count", min_value=0, value=0)
    
    # Submit button
    submitted = st.form_submit_button("🔍 Analyze Sentiment", type="primary")

# Process form submission
if submitted:
    if not text_content.strip():
        st.error("Please enter some text to analyze.")
    elif len(text_content) < 5:
        st.error("Please enter at least 5 characters for analysis.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = preprocessor.predict_sentiment(
                text_content=text_content,
                platform=platform,
                location=location,
                language=language,
                day_of_week=day_of_week,
                mentions=mentions,
                hashtags=hashtags,
                brand_name=brand_name,
                product_name=product_name,
                campaign_name=campaign_name
            )
        
        if 'error' in result:
            st.error(f"Prediction error: {result['error']}")
        else:
            # Display results
            st.success("Analysis Complete!")
            
            # Results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sentiment Score", f"{result['sentiment_score']:.4f}")
            
            with col2:
                sentiment_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
                st.metric("Sentiment", f"{sentiment_color.get(result['sentiment_label'], '⚪')} {result['sentiment_label']}")
            
            with col3:
                st.metric("Confidence", f"{result['confidence']:.4f}")
            
            # Detailed results
            st.subheader("📊 Detailed Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({
                    "Sentiment Score": result['sentiment_score'],
                    "Sentiment Label": result['sentiment_label'],
                    "Emotion": result['emotion'],
                    "Confidence": result['confidence']
                })
            
            with col2:
                st.json(result['model_info'])
            
            # Progress bar for sentiment score
            st.subheader("Sentiment Scale")
            progress_color = "normal"
            if result['sentiment_score'] > 0.6:
                progress_color = "normal"  # Green
            elif result['sentiment_score'] < 0.4:
                progress_color = "normal"  # Will show as red due to low value
            
            st.progress(result['sentiment_score'])
            st.caption("0.0 = Very Negative, 0.5 = Neutral, 1.0 = Very Positive")

# About section
st.markdown("---")
with st.expander("ℹ️ About This App"):
    st.markdown("""
    This sentiment analysis app uses a Ridge Regression model trained on social media data.
    
    **Features:**
    - Text preprocessing with NLTK
    - TF-IDF vectorization
    - Multi-feature sentiment prediction
    - Real-time analysis
    
    **Model Performance:**
    - Algorithm: Ridge Regression with L2 regularization
    - Cross-validation: 5-fold CV
    - Features: TF-IDF vectors + metadata features
    """)