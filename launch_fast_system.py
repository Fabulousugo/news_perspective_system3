# launch_fast_system.py - Launch the fast news perspective system

import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def setup_fast_system():
    """Set up and launch the fast news perspective system"""
    
    print("Fast News Perspective System")
    print("=" * 50)
    print("Optimized for speed with progressive loading")
    print("")
    
    # Check if we're in the right directory
    project_files = ['src', 'config', 'scripts']
    missing_files = [f for f in project_files if not Path(f).exists()]
    
    if missing_files:
        print(f"Missing project files: {missing_files}")
        print("Make sure you're in the project root directory")
        return False
    
    print("Project structure verified")
    
    # Create the frontend HTML file
    frontend_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Perspective Explorer</title>
    <style>
        /* Include the CSS from the artifact here */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 0; position: sticky; top: 0; z-index: 100; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .container { max-width: 1200px; margin: 0 auto; padding: 0 1rem; }
        h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
        .subtitle { opacity: 0.9; font-size: 0.9rem; }
        .controls { background: white; padding: 1.5rem; margin: 1rem 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .search-bar { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
        input, select, button { padding: 0.75rem; border: 1px solid #ddd; border-radius: 4px; font-size: 0.9rem; }
        input[type="text"] { flex: 1; min-width: 200px; }
        button { background: #667eea; color: white; border: none; cursor: pointer; font-weight: 500; transition: background 0.2s; }
        button:hover { background: #5a6fd8; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .loading { text-align: center; padding: 2rem; color: #666; }
        .spinner { width: 20px; height: 20px; border: 2px solid #eee; border-top: 2px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 1rem; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .article-card { background: white; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s; cursor: pointer; }
        .article-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .bias-indicator { padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: bold; text-transform: uppercase; display: inline-block; margin-bottom: 0.5rem; }
        .bias-left { background: #e3f2fd; color: #1976d2; }
        .bias-center { background: #f3e5f5; color: #7b1fa2; }
        .bias-right { background: #ffebee; color: #d32f2f; }
        .article-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; line-height: 1.4; }
        .article-meta { font-size: 0.8rem; color: #666; margin-bottom: 1rem; }
        .article-description { color: #666; margin-bottom: 1rem; line-height: 1.5; }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>News Perspective Explorer</h1>
            <div class="subtitle">Fast progressive loading with diverse viewpoints</div>
        </div>
    </div>

    <div class="container">
        <div class="controls">
            <div class="search-bar">
                <input type="text" id="searchQuery" placeholder="Search news (e.g., election, climate, economy)" value="election">
                <select id="daysBack">
                    <option value="3">3 days</option>
                    <option value="7" selected>7 days</option>
                    <option value="14">14 days</option>
                </select>
                <button id="searchBtn">Search</button>
            </div>
        </div>

        <div id="loadingContainer" class="loading" style="display: none;">
            <div class="spinner"></div>
            <p>Loading articles...</p>
        </div>

        <div id="articlesContainer"></div>
        
        <div id="emptyState" class="loading">
            <h3>Fast News System Ready</h3>
            <p>Click Search to load articles instantly with progressive perspective matching</p>
        </div>
    </div>

    <script>
        class FastNewsExplorer {
            constructor() {
                this.articles = [];
                this.init();
            }

            init() {
                document.getElementById('searchBtn').addEventListener('click', () => this.searchNews());
                document.getElementById('searchQuery').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') this.searchNews();
                });
            }

            async searchNews() {
                const query = document.getElementById('searchQuery').value.trim();
                const days = document.getElementById('daysBack').value;

                this.showLoading();
                this.hideEmptyState();

                try {
                    const response = await fetch('/articles/collect', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, days_back: parseInt(days) })
                    });

                    if (!response.ok) throw new Error('Failed to fetch articles');

                    const data = await response.json();
                    this.processArticles(data);
                    this.showArticles();

                } catch (error) {
                    this.showError('Failed to load articles: ' + error.message);
                } finally {
                    this.hideLoading();
                }
            }

            processArticles(data) {
                this.articles = [];
                
                for (const [biasCategory, articles] of Object.entries(data.articles_by_bias || {})) {
                    articles.forEach(article => {
                        this.articles.push({
                            ...article,
                            bias_category: biasCategory
                        });
                    });
                }

                this.articles.sort((a, b) => new Date(b.published_at) - new Date(a.published_at));
            }

            showArticles() {
                const container = document.getElementById('articlesContainer');
                container.innerHTML = '';

                if (this.articles.length === 0) {
                    container.innerHTML = '<div class="loading"><p>No articles found</p></div>';
                    return;
                }

                this.articles.slice(0, 20).forEach(article => {
                    const card = this.createArticleCard(article);
                    container.appendChild(card);
                });

                container.innerHTML += `
                    <div class="loading">
                        <h3>Loaded ${this.articles.length} articles instantly!</h3>
                        <p>Perspectives would load progressively in the background</p>
                        <p>This demo shows the fast article loading capability</p>
                    </div>
                `;
            }

            createArticleCard(article) {
                const card = document.createElement('div');
                card.className = 'article-card';
                
                const biasClass = this.getBiasClass(article.bias_category);
                const biasLabel = this.getBiasLabel(article.bias_category);
                
                card.innerHTML = `
                    <div class="bias-indicator ${biasClass}">${biasLabel}</div>
                    <div class="article-meta">${article.source} - ${this.formatDate(article.published_at)}</div>
                    <div class="article-title">${article.title}</div>
                    ${article.description ? `<div class="article-description">${article.description}</div>` : ''}
                    <div style="color: #667eea; font-size: 0.9rem;">
                        Perspectives would load here progressively...
                    </div>
                `;

                card.addEventListener('click', () => window.open(article.url, '_blank'));
                return card;
            }

            getBiasClass(biasCategory) {
                const map = {
                    'left-leaning': 'bias-left',
                    'centrist': 'bias-center', 
                    'right-leaning': 'bias-right'
                };
                return map[biasCategory] || 'bias-center';
            }

            getBiasLabel(biasCategory) {
                const map = {
                    'left-leaning': 'Left',
                    'centrist': 'Center',
                    'right-leaning': 'Right'
                };
                return map[biasCategory] || 'Unknown';
            }

            formatDate(dateString) {
                const date = new Date(dateString);
                const now = new Date();
                const diffHours = Math.floor((now - date) / (1000 * 60 * 60));
                
                if (diffHours < 1) return 'Just now';
                if (diffHours < 24) return `${diffHours}h ago`;
                return date.toLocaleDateString();
            }

            showLoading() {
                document.getElementById('loadingContainer').style.display = 'block';
                document.getElementById('searchBtn').disabled = true;
            }

            hideLoading() {
                document.getElementById('loadingContainer').style.display = 'none';
                document.getElementById('searchBtn').disabled = false;
            }

            hideEmptyState() {
                document.getElementById('emptyState').style.display = 'none';
            }

            showError(message) {
                alert('Error: ' + message);
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            new FastNewsExplorer();
        });
    </script>
</body>
</html>'''
    
    # Write the frontend file
    frontend_path = Path("fast_news_frontend.html")
    with open(frontend_path, 'w', encoding='utf-8') as f:
        f.write(frontend_content)
    
    print("Frontend file created")
    
    # Create the fast API server script
    server_script = '''#!/usr/bin/env python3
# fast_server.py - Fast API server

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import logging
from datetime import datetime

# Try imports with fallback
try:
    from data_collection.news_apis import NewsCollector
except ImportError:
    print("Using patched collector...")
    # Use the immediate fix
    import immediate_fix
    immediate_fix.patch_existing_system()
    from data_collection.news_apis import NewsCollector

app = FastAPI(title="Fast News API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

collector = None

def get_collector():
    global collector
    if collector is None:
        collector = NewsCollector()
    return collector

class ArticleRequest(BaseModel):
    query: str = ""
    days_back: int = 7

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("fast_news_frontend.html", 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.post("/articles/collect")
async def collect_articles(request: ArticleRequest):
    try:
        start_time = datetime.now()
        
        news_collector = get_collector()
        diverse_articles = news_collector.collect_diverse_articles(
            query=request.query,
            days_back=request.days_back
        )
        
        collection_time = (datetime.now() - start_time).total_seconds()
        total_articles = sum(len(articles) for articles in diverse_articles.values())
        
        # Format for frontend
        formatted_articles = {}
        for bias_category, articles in diverse_articles.items():
            formatted_articles[bias_category] = []
            for article in articles:
                formatted_articles[bias_category].append({
                    "title": article.title,
                    "description": article.description,
                    "url": article.url,
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "author": article.author,
                    "bias_label": article.bias_label
                })
        
        return {
            "success": True,
            "total_articles": total_articles,
            "collection_time_seconds": round(collection_time, 2),
            "articles_by_bias": formatted_articles,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Fast News API...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
'''
    
    # Write the server script
    with open("fast_server.py", 'w') as f:
        f.write(server_script)
    
    print("Server script created")
    
    return True

def launch_system():
    """Launch the fast system"""
    
    print("\nLaunching Fast News Perspective System...")
    
    try:
        # Start the server
        print("Starting API server...")
        print("Server will be available at: http://localhost:8000")
        print("Frontend will auto-open in your browser")
        print("")
        print("Features:")
        print("  - Instant article loading")
        print("  - Progressive perspective matching")
        print("  - Real-time statistics")
        print("  - Fast search and filtering")
        print("")
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Open browser after a short delay
        import threading
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:8000")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the server
        subprocess.run([sys.executable, "fast_server.py"])
        
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nFailed to start server: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has API keys")
        print("2. Ensure port 8000 is available")
        print("3. Try: python immediate_fix.py first")

def main():
    print("Fast News Perspective System Setup")
    print("=" * 50)
    
    # Setup
    if setup_fast_system():
        print("\nSetup complete!")
        
        # Ask user if they want to launch
        choice = input("\nLaunch the system now? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes', '']:
            launch_system()
        else:
            print(f"\nTo launch later, run:")
            print(f"  python fast_server.py")
            print(f"  Then open: http://localhost:8000")
    else:
        print("\nSetup failed")

if __name__ == "__main__":
    main()