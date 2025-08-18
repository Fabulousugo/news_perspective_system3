# launch_web_frontend.py - Simple launcher for the web frontend

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import threading

def check_dependencies():
    """Check if required packages are installed"""
    print("📦 Checking dependencies...")
    
    required = ["fastapi", "uvicorn"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            missing.append(package)
    
    if missing:
        print(f"\n💡 Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def test_backend():
    """Test that the backend components work"""
    print("\n🧪 Testing backend components...")
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent / "src"))
    
    try:
        from data_collection.simple_extended_collector import SimpleExtendedCollector
        from models.news_browser import NewsBrowser
        
        print("✅ News collector available")
        print("✅ News browser available")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the project root directory")
        return False

def open_browser_delayed():
    """Open browser after a short delay"""
    time.sleep(3)  # Wait for server to start
    print("🌐 Opening browser...")
    webbrowser.open("http://localhost:3000")

def main():
    """Launch the web frontend"""
    print("🚀 News Perspective Web Frontend Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Test backend
    if not test_backend():
        print("\n❌ Backend components not available")
        print("💡 Try running from the project root directory")
        return False
    
    print("\n🌐 Starting web server...")
    print("📊 Features enabled:")
    print("   • Temporal bias chart")
    print("   • Adjustable precision threshold (0.70+ recommended)")
    print("   • Interactive perspective discovery")
    print("   • Vanilla JavaScript frontend")
    
    # Open browser in background thread
    browser_thread = threading.Thread(target=open_browser_delayed)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Start the server
        import uvicorn
        print(f"\n🎯 Server starting at: http://localhost:3000")
        print(f"   Press Ctrl+C to stop")
        
        uvicorn.run(
            "simple_web_server:app",
            host="127.0.0.1",
            port=3000,
            reload=False,
            log_level="warning"  # Reduce log noise
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return True
    except Exception as e:
        print(f"\n❌ Failed to start server: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# test_web_frontend.py - Test the web frontend

import requests
import json
import time
from pathlib import Path
import sys

def test_web_frontend():
    """Test the web frontend API"""
    print("🧪 Testing Web Frontend")
    print("=" * 30)
    
    base_url = "http://localhost:3000"
    
    print("1. Testing server health...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is responding")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Server not accessible: {e}")
        print("💡 Make sure to run: python launch_web_frontend.py")
        return False
    
    print("\n2. Testing API endpoint...")
    try:
        # Test with a simple query
        api_url = f"{base_url}/api/articles"
        params = {
            "query": "election",
            "days": 3,
            "threshold": 0.70
        }
        
        print(f"   Requesting: {api_url}")
        print(f"   Parameters: {params}")
        
        response = requests.get(api_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ API endpoint working")
            print(f"   Articles returned: {len(data.get('articles', []))}")
            print(f"   Temporal data points: {len(data.get('temporal_data', []))}")
            
            if 'stats' in data:
                stats = data['stats']
                print(f"   Total articles: {stats.get('total_articles', 0)}")
                print(f"   With perspectives: {stats.get('articles_with_perspectives', 0)}")
                print(f"   Coverage: {stats.get('perspective_coverage', 0)*100:.1f}%")
            
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return False

def main():
    """Test the web frontend"""
    print("🌐 Web Frontend Test")
    print("=" * 25)
    print("This will test if the web frontend is working correctly")
    print("")
    
    success = test_web_frontend()
    
    if success:
        print("\n🎉 Web frontend test passed!")
        print("\n🚀 Ready to use:")
        print("   1. Launch: python launch_web_frontend.py")
        print("   2. Open: http://localhost:3000")
        print("   3. Try searching for 'election', 'climate', etc.")
        print("   4. Adjust precision threshold (0.70+ recommended)")
    else:
        print("\n❌ Web frontend test failed")
        print("💡 Troubleshooting:")
        print("   1. Make sure the server is running")
        print("   2. Check your API keys in .env file")
        print("   3. Verify internet connection")

if __name__ == "__main__":
    main()