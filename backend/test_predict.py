import sys
from predict import predict_url

def test_prediction():
    """
    Test function to verify prediction functionality.
    """
    # Test URLs
    legitimate_urls = [
        "https://www.google.com",
        "https://www.amazon.com/dp/B07PXGQC1Q/",
        "https://github.com/microsoft/vscode"
    ]
    
    suspicious_urls = [
        "http://googl3.com-secure.login.net",
        "https://paypa1-secure.com/login.php?account=verify",
        "http://banking-secure.com/login@user.verify"
    ]
    
    print("\n===== TESTING LEGITIMATE URLs =====")
    for url in legitimate_urls:
        result = predict_url(url)
        print(f"URL: {url}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']})")
        print(f"Best model: {result['best_model']}")
        print("-" * 50)
    
    print("\n===== TESTING SUSPICIOUS URLs =====")
    for url in suspicious_urls:
        result = predict_url(url)
        print(f"URL: {url}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']})")
        print(f"Best model: {result['best_model']}")
        print("-" * 50)

if __name__ == "__main__":
    print("Starting URL prediction test...")
    try:
        test_prediction()
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        sys.exit(1) 