from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
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

# Download required NLTK data
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

app = Flask(__name__)
app.secret_key = 'sentiment_analysis_app_2024'

# Load the trained model components
try:
    model_components = joblib.load('sentiment_model_ridge.joblib')
    
    ridge_model = model_components['ridge_model']
    scaler = model_components['scaler']
    tfidf_text_vectorizer = model_components['tfidf_text_vectorizer']
    tfidf_mentions_vectorizer = model_components['tfidf_mentions_vectorizer']
    tfidf_hashtags_vectorizer = model_components['tfidf_hashtags_vectorizer']
    ordinal_encoder = model_components['ordinal_encoder']
    label_encoders = model_components['label_encoders']
    feature_columns = model_components['feature_columns']
    target_column = model_components['target_column']
    
    print("✅ Model components loaded successfully!")
    print(f"📊 Model Type: {model_components.get('model_type', 'Ridge Regression')}")
    print(f"🎯 Best Alpha: {model_components['model_performance']['best_alpha']}")
    print(f"📈 R² Score: {model_components['model_performance']['r2_score']:.4f}")
    
    model_available = True
except FileNotFoundError:
    print("❌ Model file not found. Please ensure 'sentiment_model_ridge.joblib' exists.")
    model_available = False
except KeyError as e:
    print(f"❌ Error loading model components: Missing key {e}")
    model_available = False

class SentimentPreprocessor:
    def __init__(self):
        """Initialize preprocessor with same parameters as notebook"""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.regex_pattern = r'[^\w\s]|[\d]'
    
    def preprocess_text(self, text, text_type='text'):
        """Preprocess text following the exact same pipeline as in the notebook"""
        if pd.isna(text) or text == '' or text is None:
            return 'none'
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove mentions and hashtags based on text type (same logic as notebook)
        if text_type != 'mentions':
            text = re.sub(r'@\w+', '', text)
        if text_type != 'hashtags':
            text = re.sub(r'#\w+', '', text)
            text = re.sub(self.regex_pattern, ' ', text)
        else:
            text = re.sub(r'#', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenization and remove stopwords
        tokens = [word for word in word_tokenize(text) if word not in self.stop_words]
        text = ' '.join(tokens)
        
        # Lemmatization
        lemmatized = [self.lemmatizer.lemmatize(word, 'v') for word in text.split()]
        text = ' '.join(lemmatized)
        
        # Stemming
        stemmed = [self.ps.stem(word) for word in text.split()]
        text = ' '.join(stemmed)
        
        return text if text.strip() else 'none'
    
    def predict_sentiment(self, text_content, platform='Twitter', location='USA', 
                         language='English', day_of_week='Monday', 
                         mentions='', hashtags='', brand_name='Unknown', 
                         product_name='Unknown', campaign_name='Unknown', 
                         campaign_phase='Launch', likes_count=0, shares_count=0, 
                         comments_count=0, impressions=1000, engagement_rate=0.1,
                         toxicity_score=0.1, user_past_sentiment_avg=0.5,
                         user_engagement_growth=0.1, buzz_change_rate=5.0):
        """Predict sentiment using the trained Ridge model following notebook implementation"""
        
        if not model_available:
            return {"error": "Model components not loaded"}
        
        try:
            # Create input dataframe with same structure as training data
            from datetime import datetime
            current_time = datetime.now()
            
            input_data = {
                'day_of_week': [day_of_week],
                'platform': [platform], 
                'location': [location],
                'language': [language],
                'text_content': [text_content],
                'hashtags': [hashtags if hashtags else ''],
                'mentions': [mentions if mentions else 'None'],
                'sentiment_score': [0.0],  # Placeholder - will be predicted
                'sentiment_label': ['Neutral'],  # Placeholder
                'emotion_type': ['Happy'],  # Placeholder  
                'toxicity_score': [toxicity_score],
                'likes_count': [likes_count],
                'shares_count': [shares_count], 
                'comments_count': [comments_count],
                'impressions': [impressions],
                'engagement_rate': [engagement_rate],
                'brand_name': [brand_name],
                'product_name': [product_name],
                'campaign_name': [campaign_name],
                'campaign_phase': [campaign_phase],
                'user_past_sentiment_avg': [user_past_sentiment_avg],
                'user_engagement_growth': [user_engagement_growth],
                'buzz_change_rate': [buzz_change_rate],
                'year': [current_time.year],
                'month': [current_time.month],
                'day': [current_time.day],
                'hour': [current_time.hour]
            }
            
            df_input = pd.DataFrame(input_data)
            
            # Apply text preprocessing (same as notebook)
            word_columns = ['text_content', 'mentions', 'hashtags']
            for col in word_columns:
                df_input[col] = df_input[col].apply(lambda x: self.preprocess_text(x, col.replace('_content', '')))
            
            # Apply TF-IDF transformation using pre-fitted vectorizers
            tfidf_text_transformed = tfidf_text_vectorizer.transform(df_input["text_content"])
            tfidf_mentions_transformed = tfidf_mentions_vectorizer.transform(df_input["mentions"])
            tfidf_hashtags_transformed = tfidf_hashtags_vectorizer.transform(df_input["hashtags"])
            
            X_tfidf_transformed = hstack([tfidf_text_transformed, tfidf_mentions_transformed, tfidf_hashtags_transformed])
            
            # Create DataFrame from TF-IDF results
            tfidf_feature_names = (
                [f"text_{w}" for w in tfidf_text_vectorizer.get_feature_names_out()] +
                [f"mentions_{w}" for w in tfidf_mentions_vectorizer.get_feature_names_out()] +
                [f"hashtag_{w}" for w in tfidf_hashtags_vectorizer.get_feature_names_out()]
            )
            df_tfidf_transformed = pd.DataFrame.sparse.from_spmatrix(X_tfidf_transformed, columns=tfidf_feature_names)
            
            # Drop text columns (as in notebook)
            df_processed = df_input.drop(columns=['text_content', 'mentions', 'hashtags'])
            
            # Apply ordinal encoding
            ordinal_columns = ['day_of_week', 'sentiment_label', 'emotion_type', 'campaign_phase']
            df_processed[ordinal_columns] = ordinal_encoder.transform(df_processed[ordinal_columns]).astype(int)
            
            # Apply label encoding for binary columns
            for col, encoder in label_encoders.items():
                if col in df_processed.columns:
                    df_processed[col] = encoder.transform(df_processed[col])
            
            # Apply one-hot encoding
            ohe_columns = [col for col in df_processed.columns if df_processed[col].dtype == 'object']
            if ohe_columns:
                df_processed_ohe = pd.get_dummies(df_processed, columns=ohe_columns)
            else:
                df_processed_ohe = df_processed
            
            # Merge with TF-IDF features
            df_final = pd.concat([df_processed_ohe, df_tfidf_transformed], axis=1)
            
            # Align columns with training data
            for col in feature_columns:
                if col not in df_final.columns:
                    df_final[col] = 0
            
            # Remove extra columns and ensure correct order
            extra_cols = [col for col in df_final.columns if col not in feature_columns]
            df_final = df_final.drop(columns=extra_cols)
            df_final = df_final[feature_columns]
            
            # Apply scaling
            scaled_data = scaler.transform(df_final)
            
            # Remove target column for prediction
            target_column_index = feature_columns.index(target_column)
            X_scaled = np.delete(scaled_data, target_column_index, axis=1)
            
            # Make prediction using Ridge model
            predicted_score = ridge_model.predict(X_scaled)[0]
            
            # Convert back to original scale (scaled sentiment score back to 0-1)
            # Note: May need adjustment based on actual scaling
            predicted_score = max(0.0, min(1.0, predicted_score))
            
            # Determine sentiment label and emotion
            if predicted_score > 0.6:
                sentiment_label = "Positive"
                emotion = "Happy" 
            elif predicted_score < 0.4:
                sentiment_label = "Negative"
                emotion = "Sad"
            else:
                sentiment_label = "Neutral"
                emotion = "Neutral"
            
            return {
                "sentiment_score": round(predicted_score, 4),
                "sentiment_label": sentiment_label,
                "emotion": emotion,
                "confidence": round(abs(predicted_score - 0.5) * 2, 4),
                "model_info": {
                    "model_type": "Ridge Regression",
                    "alpha": ridge_model.alpha_,
                    "features_used": len(feature_columns) - 1  # Exclude target column
                }
            }
            
        except Exception as e:
            import traceback
            return {"error": f"Prediction error: {str(e)}", "traceback": traceback.format_exc()}

# Initialize preprocessor
preprocessor = SentimentPreprocessor()

@app.route('/')
def index():
    """Home page with sentiment analysis form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle sentiment prediction requests"""
    try:
        # Get form data
        text_content = request.form.get('text_content', '').strip()
        platform = request.form.get('platform', 'Twitter')
        location = request.form.get('location', 'USA')
        language = request.form.get('language', 'English')
        day_of_week = request.form.get('day_of_week', 'Monday')
        mentions = request.form.get('mentions', '')
        hashtags = request.form.get('hashtags', '')
        
        # Validate input
        if not text_content:
            flash('Please enter some text to analyze.', 'error')
            return redirect(url_for('index'))
        
        if len(text_content) < 5:
            flash('Please enter at least 5 characters for analysis.', 'error')
            return redirect(url_for('index'))
        
        # Make prediction
        result = preprocessor.predict_sentiment(
            text_content=text_content,
            platform=platform,
            location=location,
            language=language,
            day_of_week=day_of_week,
            mentions=mentions,
            hashtags=hashtags
        )
        
        if 'error' in result:
            flash(f'Prediction error: {result["error"]}', 'error')
            return redirect(url_for('index'))
        
        return render_template('result.html', 
                             result=result, 
                             input_data={
                                 'text_content': text_content,
                                 'platform': platform,
                                 'location': location,
                                 'language': language,
                                 'day_of_week': day_of_week,
                                 'mentions': mentions,
                                 'hashtags': hashtags
                             })
        
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for sentiment prediction"""
    try:
        data = request.get_json()
        
        if not data or 'text_content' not in data:
            return jsonify({'error': 'Missing text_content in request'}), 400
        
        text_content = data.get('text_content', '').strip()
        if not text_content:
            return jsonify({'error': 'text_content cannot be empty'}), 400
        
        result = preprocessor.predict_sentiment(
            text_content=text_content,
            platform=data.get('platform', 'Twitter'),
            location=data.get('location', 'USA'),
            language=data.get('language', 'English'),
            day_of_week=data.get('day_of_week', 'Monday'),
            mentions=data.get('mentions', ''),
            hashtags=data.get('hashtags', '')
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page with model information"""
    if model_available:
        model_info = {
            'model_type': model_components.get('model_type', 'Ridge Regression'),
            'features': 'TF-IDF vectors (5000 features) from text, mentions, hashtags + metadata features',
            'performance': model_components['model_performance'],
            'dataset_info': model_components['dataset_info'],
            'creation_date': model_components['creation_date'],
            'preprocessing': [
                'Text cleaning and normalization',
                'Stop word removal', 
                'Lemmatization and stemming',
                'TF-IDF vectorization (text: 3000, mentions: 1000, hashtags: 1000)',
                'Ordinal encoding for categorical features',
                'One-hot encoding for multi-category features',
                'Standard scaling for all features'
            ],
            'model_details': {
                'algorithm': 'Ridge Regression with L2 regularization',
                'alpha': model_components['model_performance']['best_alpha'],
                'cross_validation': '5-fold CV',
                'feature_count': model_components['dataset_info']['features_count']
            }
        }
    else:
        model_info = {
            'error': 'Model not loaded',
            'status': 'Model components could not be loaded from sentiment_model_ridge.joblib'
        }
    
    return render_template('about.html', model_info=model_info)

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sentiment Analysis Flask App')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the app on (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    print("🚀 Starting Sentiment Analysis Flask App...")
    print("📊 Model Status:", "✅ Loaded" if model_available else "❌ Not Found")
    if model_available:
        print(f"🤖 Model Type: {model_components.get('model_type', 'Ridge Regression')}")
        print(f"🎯 Best Alpha: {model_components['model_performance']['best_alpha']}")
        print(f"📈 R² Score: {model_components['model_performance']['r2_score']:.4f}")
    print(f"🌐 Server will start at: http://{args.host}:{args.port}")
    print("📝 Press CTRL+C to stop the server")
    print("-" * 50)
    
    try:
        # Try to start the Flask app with safer defaults
        app.run(
            debug=args.debug, 
            host=args.host, 
            port=args.port, 
            threaded=True,
            use_reloader=False,  # Disable reloader to prevent crashes
            use_debugger=args.debug
        )
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {args.port} is already in use!")
            print("💡 Solutions:")
            print(f"   1. Use a different port: python3 app.py --port {args.port + 1}")
            print(f"   2. Kill the process using port {args.port}:")
            print(f"      lsof -ti:{args.port} | xargs kill -9")
        else:
            print(f"❌ Network error: {e}")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
