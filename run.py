#!/usr/bin/env python3
"""
Production-ready runner for Sentiment Analysis Flask App
"""

import os
import sys

def main():
    """Run the Flask app in production mode"""
    print("🚀 Starting Sentiment Analysis Flask App (Production Mode)")
    print("=" * 60)
    
    # Set environment variables for production
    os.environ['FLASK_ENV'] = 'production'
    os.environ['FLASK_DEBUG'] = '0'
    
    try:
        from app import app
        print("✅ App imported successfully")
        print("🌐 Starting server at http://localhost:5000")
        print("📝 Press CTRL+C to stop")
        print("-" * 50)
        
        # Run in production mode - no debug, no reloader
        app.run(
            host='localhost',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
