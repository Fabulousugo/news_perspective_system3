# scripts/run_frontend.py - Simple script to run the frontend server

import sys
import uvicorn
from pathlib import Path
import logging

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

def create_frontend_directory():
    """Create frontend directory and save the HTML file"""
    frontend_dir = Path("frontend")
    frontend_dir.mkdir(exist_ok=True)
    
    # The HTML content will be served directly by the API
    # This just ensures the directory exists
    
    print(f"Frontend directory ready: {frontend_dir.absolute()}")
    return frontend_dir

def run_server(host="127.0.0.1", port=8000):
    """Run the frontend server"""
    
    print("Starting News Perspective Frontend Server")
    print("=" * 50)
    
    # Create frontend directory
    create_frontend_directory()
    
    # Import and run the API
    try:
        from src.api.simple_routes import app
        
        print(f"Starting server...")
        print(f"Frontend URL: http://{host}:{port}")
        print(f"API Docs: http://{host}:{port}/docs")
        print(f"Health Check: http://{host}:{port}/api/health")
        print("")
        print("Features:")
        print("   Interactive threshold adjustment")
        print("   Temporal bias visualization")
        print("   Real-time perspective matching")
        print("   Mobile-friendly design")
        print("")
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        uvicorn.run(
            app, 
            host=host, 
            port=port, 
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project root directory")
        print("2. Check that src/api/simple_routes.py exists")
        print("3. Verify your Python path includes the src directory")
        return False
        
    except Exception as e:
        print(f"Server startup failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the News Perspective Frontend")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    run_server(args.host, args.port)

# save_frontend.py - Script to save the HTML frontend to a file

def save_frontend_html():
    """Save the frontend HTML to a file for standalone use"""
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Perspective Browser</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <!-- CSS and JavaScript content from the artifact above -->
</head>
<body>
    <!-- Same body content as above -->
</body>
</html>'''
    
    frontend_dir = Path("frontend")
    frontend_dir.mkdir(exist_ok=True)
    
    html_file = frontend_dir / "index.html"
    
    # For now, create a simple redirect page
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
    
    print(f"Frontend HTML saved to: {html_file}")
    return html_file

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save_frontend_html()
    else:
        run_server()
