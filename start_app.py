#!/usr/bin/env python3
"""
Simple startup script for the Sentiment Analysis Flask App
This script helps debug startup issues
"""

import sys
import os
import traceback

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    # Check required modules
    required_modules = [
        'flask', 'sklearn', 'nltk', 'joblib', 
        'pandas', 'numpy', 'scipy'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    # Check model file
    model_path = 'sentiment_model_ridge.joblib'
    if os.path.exists(model_path):
        print(f"✅ Model file found: {model_path}")
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("   The app will still run but predictions may not work properly")
    
    # Check template and static directories
    if os.path.exists('templates'):
        print("✅ Templates directory found")
    else:
        print("❌ Templates directory missing")
        return False
        
    if os.path.exists('static'):
        print("✅ Static directory found")
    else:
        print("❌ Static directory missing")
        return False
    
    return True

def start_app():
    """Start the Flask application"""
    print("\n🚀 Starting Flask application...")
    
    try:
        from app import app
        print("✅ App module imported successfully")
        
        # Start the app with basic configuration
        print("🌐 Starting server on http://localhost:5001")
        print("📝 Press CTRL+C to stop")
        print("-" * 50)
        
        app.run(
            debug=True,
            host='localhost',  # Use localhost instead of 0.0.0.0 for debugging
            port=5001,         # Use different port to avoid conflicts
            threaded=True,
            use_reloader=False  # Disable reloader to avoid double startup
        )
        
    except ImportError as e:
        print(f"❌ Failed to import app: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to start app: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🧠 Sentiment Analysis Flask App - Debug Startup")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    print("\n✅ Environment check passed!")
    
    # Start the application
    try:
        start_app()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()
