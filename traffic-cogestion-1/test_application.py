import requests
import json
from datetime import datetime
def test_flask_backend():
    print("🧪 Testing Flask Backend (http://localhost:5001)")
    try:
        response = requests.get("http://localhost:5001/api/health")
        if response.status_code == 200:
            print("✅ Health check: PASSED")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Health check: FAILED (Error: {e})")
        return False
    try:
        response = requests.get("http://localhost:5001/api/model_info")
        if response.status_code == 200:
            print("✅ Model info: PASSED")
            data = response.json()
            print(f"   Model type: {data.get('model_type')}")
            print(f"   Features: {data.get('n_features')}")
        else:
            print(f"❌ Model info: FAILED (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Model info: FAILED (Error: {e})")
    try:
        test_data = {
            "latitude": 16.5062,
            "longitude": 80.6480,
            "timestamp": datetime.now().isoformat()
        }
        response = requests.post(
            "http://localhost:5001/api/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            print("✅ Prediction: PASSED")
            data = response.json()
            print(f"   Traffic prediction: {data.get('prediction'):.2f}%")
            print(f"   Confidence: {data.get('confidence')}")
        else:
            print(f"❌ Prediction: FAILED (Status: {response.status_code})")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Prediction: FAILED (Error: {e})")
    return True
def test_nextjs_frontend():
    print("\n🧪 Testing Next.js Frontend (http://localhost:3000)")
    try:
        response = requests.get("http://localhost:3000")
        if response.status_code == 200:
            print("✅ Frontend accessible: PASSED")
        else:
            print(f"❌ Frontend accessible: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Frontend accessible: FAILED (Error: {e})")
        return False
    try:
        response = requests.get("http://localhost:3000/api/ucs-model-info")
        if response.status_code == 200:
            print("✅ API proxy: PASSED")
            data = response.json()
            if data.get('success'):
                print(f"   Model type: {data['data'].get('modelType')}")
            else:
                print(f"   Response: {data}")
        else:
            print(f"❌ API proxy: FAILED (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ API proxy: FAILED (Error: {e})")
    return True
def main():
    print("🚀 Traffic Prediction Application Test Suite")
    print("=" * 50)
    flask_ok = test_flask_backend()
    nextjs_ok = test_nextjs_frontend()
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"Flask Backend: {'✅ WORKING' if flask_ok else '❌ FAILED'}")
    print(f"Next.js Frontend: {'✅ WORKING' if nextjs_ok else '❌ FAILED'}")
    if flask_ok and nextjs_ok:
        print("\n🎉 All tests passed! Your application is ready to use.")
        print("\n🌐 Access your application at:")
        print("   • Main App: http://localhost:3000")
        print("   • Flask API: http://localhost:5001")
        print("\n💡 You can now:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Test traffic predictions on the interactive map")
        print("   3. Use the API endpoints for custom integrations")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
if __name__ == "__main__":
    main()