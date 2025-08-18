# setup_frontend.py - Quick frontend setup script

import os
import webbrowser
import subprocess
import time
from pathlib import Path

def create_frontend_files():
    """Create frontend directory and files"""
    
    print("🌐 Setting up News Perspective Frontend")
    print("=" * 50)
    
    # Create frontend directory
    frontend_dir = Path("frontend")
    frontend_dir.mkdir(exist_ok=True)
    
    print(f"✅ Created frontend directory: {frontend_dir.absolute()}")
    
    # Create simple HTML file
    simple_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Perspectives</title>
    <style>
        body { font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; padding: 2rem; text-align: center; border-radius: 12px; margin-bottom: 1rem; }
        .search-form { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem; }
        .form-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
        .form-group { flex: 1; }
        .form-group label { display: block; margin-bottom: 0.5rem; font-weight: 600; }
        .form-group input, .form-group select { width: 100%; padding: 0.75rem; border: 2px solid #e5e7eb; border-radius: 8px; }
        .btn { background: #4f46e5; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; cursor: pointer; width: 100%; font-size: 1rem; font-weight: 600; }
        .btn:hover { background: #4338ca; }
        .loading { text-align: center; padding: 2rem; display: none; }
        .spinner { width: 32px; height: 32px; border: 3px solid #e5e7eb; border-top: 3px solid #4f46e5; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 1rem; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .article { background: white; padding: 1.5rem; margin-bottom: 1rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #4f46e5; }
        .article-title { font-weight: 700; margin-bottom: 0.5rem; }
        .article-source { color: #6b7280; margin-bottom: 1rem; }
        .perspectives { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #f3f4f6; }
        .perspective-item { background: #f9fafb; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; }
        .error { background: #fee2e2; color: #dc2626; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; display: none; }
        @media (max-width: 640px) { .form-row { flex-direction: column; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📰 News Perspectives</h1>
            <p>Compare viewpoints across the political spectrum</p>
        </div>

        <div class="search-form">
            <form id="searchForm">
                <div class="form-row">
                    <div class="form-group">
                        <label for="query">Search Topic</label>
                        <input type="text" id="query" placeholder="e.g., election, climate, economy">
                    </div>
                    <div class="form-group">
                        <label for="days">Time Range</label>
                        <select id="days">
                            <option value="3">3 days</option>
                            <option value="7" selected>1 week</option>
                            <option value="14">2 weeks</option>
                        </select>
                    </div>
                </div>
                <button type="submit" class="btn">🔍 Find Perspectives</button>
            </form>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Finding perspective matches...</p>
        </div>

        <div class="error" id="error">
            <p id="errorText"></p>
        </div>

        <div id="results"></div>
    </div>

    <script>
        const API_URL = 'http://localhost:8000';
        
        document.getElementById('searchForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const query = document.getElementById('query').value;
            const days = parseInt(document.getElementById('days').value);
            
            showLoading(true);
            hideError();
            
            try {
                const response = await fetch(`${API_URL}/perspectives/find`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, days_back: days, min_perspectives: 2 })
                });
                
                if (!response.ok) throw new Error(`Error: ${response.status}`);
                
                const data = await response.json();
                displayResults(data.matches || []);
                
            } catch (error) {
                showError('Failed to connect to API. Make sure the server is running on localhost:8000');
                console.error(error);
            } finally {
                showLoading(false);
            }
        });
        
        function displayResults(matches) {
            const resultsDiv = document.getElementById('results');
            
            if (matches.length === 0) {
                resultsDiv.innerHTML = '<div style="text-align: center; padding: 2rem; color: #6b7280;"><h3>No matches found</h3><p>Try a different search term</p></div>';
                return;
            }
            
            resultsDiv.innerHTML = matches.map(match => {
                const perspectives = Object.entries(match.perspectives || {});
                if (perspectives.length === 0) return '';
                
                const [mainBias, mainArticle] = perspectives[0];
                const otherPerspectives = perspectives.slice(1);
                
                return `
                    <div class="article">
                        <h3 class="article-title">${escapeHtml(mainArticle.title)}</h3>
                        <div class="article-source">📰 ${escapeHtml(mainArticle.source)} | Confidence: ${Math.round(match.confidence * 100)}%</div>
                        
                        ${otherPerspectives.length > 0 ? `
                            <div class="perspectives">
                                <strong>Other Perspectives:</strong>
                                ${otherPerspectives.map(([bias, article]) => `
                                    <div class="perspective-item">
                                        <strong>${escapeHtml(article.title)}</strong><br>
                                        📰 ${escapeHtml(article.source)}
                                    </div>
                                `).join('')}
                            </div>
                        ` : ''}
                    </div>
                `;
            }).join('');
        }
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        function showError(message) {
            document.getElementById('errorText').textContent = message;
            document.getElementById('error').style.display = 'block';
        }
        
        function hideError() {
            document.getElementById('error').style.display = 'none';
        }
        
        function escapeHtml(unsafe) {
            return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }
        
        // Load initial data
        document.getElementById('query').value = 'election';
        document.getElementById('searchForm').dispatchEvent(new Event('submit'));
    </script>
</body>
</html>'''
    
    # Write HTML file
    html_file = frontend_dir / "index.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(simple_html)
    
    print(f"✅ Created index.html: {html_file.absolute()}")
    
    return frontend_dir

def start_server(frontend_dir):
    """Start a simple HTTP server for the frontend"""
    
    print(f"\n🚀 Starting HTTP server...")
    
    try:
        # Change to frontend directory
        os.chdir(frontend_dir)
        
        # Start Python HTTP server
        port = 3000
        print(f"📡 Server starting on http://localhost:{port}")
        print(f"📱 Frontend will open automatically...")
        
        # Start server in background
        server_process = subprocess.Popen([
            'python', '-m', 'http.server', str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Open browser
        webbrowser.open(f'http://localhost:{port}')
        
        print(f"\n✅ Frontend is running!")
        print(f"🌐 URL: http://localhost:{port}")
        print(f"📂 Directory: {frontend_dir.absolute()}")
        print(f"\n📋 Instructions:")
        print(f"   1. Make sure your API server is running:")
        print(f"      python scripts/run_application.py serve")
        print(f"   2. The frontend should work automatically")
        print(f"   3. Press Ctrl+C to stop the server")
        
        # Wait for server
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping server...")
            server_process.terminate()
            print(f"✅ Server stopped")
        
    except FileNotFoundError:
        print(f"❌ Python not found in PATH")
        print(f"💡 Alternative: Open {frontend_dir / 'index.html'} directly in your browser")
        return False
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        print(f"💡 Alternative: Open {frontend_dir / 'index.html'} directly in your browser")
        return False

def check_api_server():
    """Check if the API server is running"""
    
    print(f"🔍 Checking API server...")
    
    try:
        import requests
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            print(f"✅ API server is running on http://localhost:8000")
            return True
        else:
            print(f"⚠️  API server responded with status {response.status_code}")
            return False
    except ImportError:
        print(f"⚠️  requests library not installed, cannot check API")
        return None
    except Exception as e:
        print(f"❌ API server not responding: {e}")
        print(f"💡 Start it with: python scripts/run_application.py serve")
        return False

def main():
    """Main setup function"""
    
    print("🌐 News Perspective Frontend Setup")
    print("=" * 50)
    
    # Check API server
    api_status = check_api_server()
    
    # Create frontend files
    frontend_dir = create_frontend_files()
    
    # Ask user what to do
    print(f"\n🎯 Setup Options:")
    print(f"1. Start HTTP server (recommended)")
    print(f"2. Just create files (manual setup)")
    
    if api_status is False:
        print(f"\n⚠️  Note: API server is not running")
        print(f"   Start it first: python scripts/run_application.py serve")
    
    try:
        choice = input(f"\nChoose option (1 or 2): ").strip()
        
        if choice == "1":
            start_server(frontend_dir)
        else:
            print(f"\n✅ Files created in: {frontend_dir.absolute()}")
            print(f"\n📋 Manual setup:")
            print(f"   1. Start your API server: python scripts/run_application.py serve")
            print(f"   2. Open {frontend_dir / 'index.html'} in your browser")
            print(f"   3. Or use a local server like: python -m http.server 3000")
        
    except KeyboardInterrupt:
        print(f"\n\n🛑 Setup cancelled")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")

if __name__ == "__main__":
    main()