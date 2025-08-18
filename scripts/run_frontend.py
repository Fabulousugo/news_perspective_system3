# scripts/run_frontend.py - Simple script to run the frontend server

import sys
import uvicorn
from pathlib import Path
import logging
import signal
import asyncio

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

def create_frontend_directory():
    """Create frontend directory and save the HTML file"""
    frontend_dir = Path("frontend")
    frontend_dir.mkdir(exist_ok=True)
    
    print(f"✅ Frontend directory ready: {frontend_dir.absolute()}")
    return frontend_dir

def run_server(host="127.0.0.1", port=8000):
    """Run the frontend server"""
    print("🚀 Starting News Perspective Frontend Server")
    print("=" * 50)
    
    create_frontend_directory()
    
    try:
        from src.api.simple_routes import app

        print(f"📡 Starting server...")
        print(f"🌐 Frontend URL: http://{host}:{port}")
        print(f"📚 API Docs: http://{host}:{port}/docs")
        print(f"❤️  Health Check: http://{host}:{port}/api/health")
        print("")
        print("💡 Features:")
        print("   🎯 Interactive threshold adjustment")
        print("   📊 Temporal bias visualization")
        print("   🔍 Real-time perspective matching")
        print("   📱 Mobile-friendly design")
        print("")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)

        try:
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n🛑 Frontend server stopped gracefully.")
        except Exception as e:
            print(f"❌ Server error: {e}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Check that src/api/simple_routes.py exists")
        print("3. Verify your Python path includes the src directory")
        return False

def save_frontend_html():
    """Save the frontend HTML to a file for standalone use"""
    frontend_dir = Path("frontend")
    frontend_dir.mkdir(exist_ok=True)
    html_file = frontend_dir / "index.html"
    
    simple_html = '''<!DOCTYPE html>
<html>
<head>
    <title>News Perspective Browser</title>
    <meta http-equiv="refresh" content="0; url=/">
</head>
<body>
    <h1>Redirecting to News Perspective Browser...</h1>
    <p>If you're not redirected, <a href="/">click here</a>.</p>
</body>
</html>'''
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(simple_html)
    
    print(f"✅ Frontend HTML saved to: {html_file}")
    return html_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the News Perspective Frontend")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--save", action="store_true", help="Save frontend HTML instead of running server")

    args = parser.parse_args()
    
    if args.save:
        save_frontend_html()
    else:
        run_server(args.host, args.port)
