import requests
import json

def test_api_endpoints():
    """Test all API endpoints to ensure they're working"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Agentic Document AI API...")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    print()
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            print("✅ Status check passed")
            status = response.json()
            print(f"   Pipeline status: {status['status']}")
            print(f"   Capabilities: {len(status['capabilities'])} features")
        else:
            print(f"❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status check error: {e}")
    
    print()
    
    # Test confidence endpoint
    try:
        response = requests.get(f"{base_url}/getConfidence/investor_name")
        if response.status_code == 200:
            print("✅ Confidence endpoint passed")
            confidence = response.json()
            print(f"   Field: {confidence['field_name']}")
            print(f"   Overall confidence: {confidence['overall_confidence']}")
        else:
            print(f"❌ Confidence check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Confidence check error: {e}")
    
    print()
    
    # Test reason update endpoint
    try:
        test_data = {
            "field_name": "investor_name",
            "value": "Jane Smith",
            "user_correction": True,
            "reasoning": "User corrected name"
        }
        response = requests.post(f"{base_url}/reasonUpdate", json=test_data)
        if response.status_code == 200:
            print("✅ Reason update passed")
            result = response.json()
            print(f"   Updated field: {result['field_update']['field_name']}")
            print(f"   New confidence: {result['field_update']['confidence']}")
            print(f"   Affected fields: {result['affected_fields']}")
        else:
            print(f"❌ Reason update failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Reason update error: {e}")
    
    print()
    print("🎉 API testing complete!")

if __name__ == "__main__":
    test_api_endpoints()
