#!/usr/bin/env python3
"""
Test mobile accessibility of model files
"""
import requests
import sys

def test_mobile_access():
    base_url = "http://127.0.0.1:8000"
    
    print("📱 Testing Mobile AR Model Access")
    print("=" * 50)
    
    # Test model files
    test_models = [
        "/sofas/black_sofa.glb",
        "/tables/Dinning%20Table.glb", 
        "/chairs/Recliner%20Chair.glb",
        "/cabinets/wooden_cabinet.glb"
    ]
    
    print(f"Testing models from: {base_url}")
    print()
    
    for model_path in test_models:
        model_url = f"{base_url}{model_path}"
        print(f"Testing: {model_path}")
        
        try:
            # Test HEAD request (what AR viewers typically do)
            response = requests.head(model_url, timeout=10)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', 'unknown')
                content_length = response.headers.get('content-length', 'unknown')
                print(f"  ✅ Status: {response.status_code}")
                print(f"  📄 Content-Type: {content_type}")
                print(f"  📏 Content-Length: {content_length}")
                
                # Check if it's the right content type
                if 'gltf' in content_type.lower() or 'binary' in content_type.lower():
                    print(f"  ✅ Correct content type for AR")
                else:
                    print(f"  ⚠️  Content type might not be optimal for AR")
                    
            else:
                print(f"  ❌ Status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    print("=" * 50)
    print("🎯 Mobile AR Requirements Check:")
    print("1. Models must be accessible via HTTP GET requests")
    print("2. Content-Type should be 'model/gltf-binary' or similar")
    print("3. CORS headers must allow cross-origin access")
    print("4. Server must be accessible from mobile device network")
    print()
    print("📱 To test on mobile:")
    print("1. Find your computer's IP address (e.g., 192.168.1.100)")
    print("2. Replace 127.0.0.1 with your IP in model URLs")
    print("3. Test URLs in mobile browser")
    print("4. Generate QR codes with mobile-accessible URLs")

if __name__ == "__main__":
    test_mobile_access()
