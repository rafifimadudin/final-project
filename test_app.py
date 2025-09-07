"""
Test script for the Sentiment Analysis Flask application.
"""

import requests
import json
import time

def test_api_endpoint(base_url="http://localhost:5000"):
    """
    Test the API endpoint with various inputs.
    """
    print("🧪 Testing Sentiment Analysis API")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "text_content": "Saya sangat senang dengan produk ini! Kualitasnya luar biasa dan pelayanannya memuaskan.",
            "platform": "Twitter",
            "location": "Indonesia",
            "language": "Indonesian",
            "expected_sentiment": "Positive"
        },
        {
            "text_content": "Produk ini sangat mengecewakan. Kualitas buruk dan harga mahal.",
            "platform": "Facebook",
            "location": "Indonesia", 
            "language": "Indonesian",
            "expected_sentiment": "Negative"
        },
        {
            "text_content": "Produk standar, tidak ada yang istimewa tapi juga tidak mengecewakan.",
            "platform": "Instagram",
            "location": "USA",
            "language": "English",
            "expected_sentiment": "Neutral"
        },
        {
            "text_content": "Amazing product! Love it so much, highly recommended!",
            "platform": "Twitter",
            "location": "USA",
            "language": "English",
            "expected_sentiment": "Positive"
        },
        {
            "text_content": "Terrible service, very disappointed with this company.",
            "platform": "LinkedIn",
            "location": "UK",
            "language": "English",
            "expected_sentiment": "Negative"
        }
    ]
    
    url = f"{base_url}/api/predict"
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Text: {test_case['text_content'][:50]}...")
        print(f"Expected: {test_case['expected_sentiment']}")
        
        try:
            response = requests.post(url, json=test_case, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'error' in result:
                    print(f"❌ API Error: {result['error']}")
                else:
                    sentiment_label = result.get('sentiment_label', 'Unknown')
                    sentiment_score = result.get('sentiment_score', 0)
                    confidence = result.get('confidence', 0)
                    
                    print(f"✅ Result: {sentiment_label} (Score: {sentiment_score:.4f}, Confidence: {confidence:.4f})")
                    
                    # Check if prediction matches expectation (loose matching)
                    if sentiment_label.lower() == test_case['expected_sentiment'].lower():
                        print("🎯 Prediction matches expectation!")
                        success_count += 1
                    else:
                        print("⚠️  Prediction differs from expectation")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Is the Flask app running?")
            print("💡 Start the app with: python app.py")
            break
        except requests.exceptions.Timeout:
            print("❌ Request timeout")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n📊 Summary: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)

def test_web_interface(base_url="http://localhost:5000"):
    """
    Test if the web interface is accessible.
    """
    print("\n🌐 Testing Web Interface")
    print("=" * 30)
    
    endpoints = [
        ("/", "Home Page"),
        ("/about", "About Page")
    ]
    
    for endpoint, name in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ {name}: Accessible")
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {name}: Connection failed")
            return False
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    return True

def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    print("📦 Checking Dependencies")
    print("=" * 25)
    
    required_packages = [
        'flask', 'sklearn', 'pandas', 'numpy', 
        'nltk', 'joblib', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def main():
    """
    Main testing function.
    """
    print("🚀 Sentiment Analysis App Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before testing.")
        return
    
    # Test web interface
    if not test_web_interface():
        print("\n❌ Web interface test failed.")
        print("💡 Make sure the Flask app is running with: python app.py")
        return
    
    # Test API
    if test_api_endpoint():
        print("\n🎉 All tests passed! The application is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n📝 Manual Testing:")
    print("1. Open http://localhost:5000 in your browser")
    print("2. Try analyzing different types of text")
    print("3. Check the About page for model information")
    print("\n✨ Happy testing!")

if __name__ == "__main__":
    main()
