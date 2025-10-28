#!/usr/bin/env python3
"""
Quick test script to verify backend fixes
"""
import requests
import json

def test_backend():
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Testing Backend Fixes")
    print("=" * 50)
    
    # Test 1: Backend connectivity
    try:
        response = requests.get(f"{base_url}/get-initial-furniture")
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print(f"❌ Backend error: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Backend not accessible: {e}")
        return
    
    # Test 2: Test each category
    categories = ['sofas', 'tables', 'chairs', 'cabinets']
    
    for category in categories:
        print(f"\n📁 Testing {category.upper()} category:")
        
        try:
            # Select category
            select_response = requests.post(f"{base_url}/select-category", 
                                          json={"category": category})
            if select_response.status_code != 200:
                print(f"  ❌ Failed to select category: {select_response.status_code}")
                continue
            
            # Get furniture data
            furniture_response = requests.get(f"{base_url}/get-initial-furniture")
            if furniture_response.status_code != 200:
                print(f"  ❌ Failed to get furniture data: {furniture_response.status_code}")
                continue
            
            data = furniture_response.json()
            inventory = data.get('inventory', [])
            
            print(f"  ✅ Category loaded: {len(inventory)} items")
            
            # Test each furniture item
            for item in inventory:
                filename = item.get('filename', '')
                model_url = item.get('glb_model', '')
                thumbnail_url = item.get('thumbnail', '')
                
                print(f"    🔍 {filename}:")
                
                # Test model file
                try:
                    model_response = requests.head(model_url, timeout=5)
                    if model_response.status_code == 200:
                        print(f"      ✅ Model: {model_url}")
                    else:
                        print(f"      ❌ Model: {model_url} (Status: {model_response.status_code})")
                except Exception as e:
                    print(f"      ❌ Model: {model_url} (Error: {e})")
                
                # Test thumbnail
                try:
                    thumb_response = requests.head(thumbnail_url, timeout=5)
                    if thumb_response.status_code == 200:
                        print(f"      ✅ Thumbnail: {thumbnail_url}")
                    else:
                        print(f"      ❌ Thumbnail: {thumbnail_url} (Status: {thumb_response.status_code})")
                except Exception as e:
                    print(f"      ❌ Thumbnail: {thumbnail_url} (Error: {e})")
        
        except Exception as e:
            print(f"  ❌ Category test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_backend()
