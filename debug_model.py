import joblib
import pandas as pd

def debug_model_file():
    try:
        # Load model file
        model_components = joblib.load('sentiment_model_ridge (1).joblib')
        
        print("🔍 MODEL FILE STRUCTURE:")
        print("=" * 50)
        
        # Show all keys
        print(f"📋 Available keys: {list(model_components.keys())}")
        print()
        
        # Show details for each key
        for key, value in model_components.items():
            print(f"🔑 Key: '{key}'")
            print(f"   Type: {type(value)}")
            
            # Check if it's a model (has predict method)
            if hasattr(value, 'predict'):
                print(f"   ✅ This is a model/predictor")
            
            # Check if it's a vectorizer
            if hasattr(value, 'transform'):
                print(f"   🔄 This is a transformer/vectorizer")
                
            # Show shape if applicable
            if hasattr(value, 'shape'):
                print(f"   📏 Shape: {value.shape}")
                
            print()
            
        return model_components
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

if __name__ == "__main__":
    debug_model_file()