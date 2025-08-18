# simple_web_server.py - Simple web server for news perspective frontend

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Simple imports to avoid ASGI errors
try:
    from data_collection.simple_extended_collector import SimpleExtendedCollector
    from models.news_browser import NewsBrowser
except ImportError:
    print("❌ Import error. Using fallback system.")
    SimpleExtendedCollector = None
    NewsBrowser = None

# Simple FastAPI app
app = FastAPI(title="News Perspective Frontend", description="Simple frontend for news perspective analysis")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialize once)
collector = None
browser = None

def initialize_components():
    """Initialize news collection components"""
    global collector, browser
    if SimpleExtendedCollector and NewsBrowser:
        try:
            collector = SimpleExtendedCollector()
            browser = NewsBrowser()
            print("✅ Components initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            return False
    return False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_components()

# Serve static files
static_dir = Path("web_frontend")
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend page"""
    return get_frontend_html()

@app.get("/api/articles")
async def get_articles(query: str = "", days: int = 7, threshold: float = 0.70):
    """Get articles with perspectives"""
    
    if not collector or not browser:
        return JSONResponse({
            "error": "System not initialized. Check your configuration.",
            "articles": [],
            "temporal_data": []
        })
    
    try:
        # Update similarity threshold for better precision
        import sys
        sys.path.append(str(Path(__file__).parent / "src"))
        from config.settings import settings
        settings.SIMILARITY_THRESHOLD = threshold
        
        print(f"🔍 Collecting articles: query='{query}', days={days}, threshold={threshold}")
        
        # Collect diverse articles
        diverse_articles = collector.collect_diverse_articles(
            query=query,
            days_back=days
        )
        
        # Flatten articles
        all_articles = []
        for bias_category, articles in diverse_articles.items():
            all_articles.extend(articles)
        
        if not all_articles:
            return JSONResponse({
                "message": "No articles found",
                "articles": [],
                "temporal_data": []
            })
        
        # Browse articles for perspectives
        browseable_articles = browser.browse_articles(all_articles, sort_by="diverse")
        
        # Format articles for frontend
        formatted_articles = []
        temporal_data = []
        
        for browseable in browseable_articles[:50]:  # Limit for performance
            article = browseable.article
            
            # Format article
            formatted_article = {
                "id": hash(article.url) % 1000000,
                "title": article.title,
                "source": article.source,
                "url": article.url,
                "published_at": article.published_at.isoformat(),
                "bias_label": article.bias_label,
                "bias_name": get_bias_name(article.bias_label),
                "perspective_count": browseable.perspective_count,
                "topic": browseable.topic_cluster,
                "description": article.description or "",
                "perspectives": []
            }
            
            # Add perspectives
            for related_article, similarity in browseable.related_articles:
                perspective = {
                    "title": related_article.title,
                    "source": related_article.source,
                    "url": related_article.url,
                    "bias_label": related_article.bias_label,
                    "bias_name": get_bias_name(related_article.bias_label),
                    "similarity": similarity,
                    "published_at": related_article.published_at.isoformat()
                }
                formatted_article["perspectives"].append(perspective)
            
            formatted_articles.append(formatted_article)
            
            # Add to temporal data
            temporal_data.append({
                "date": article.published_at.isoformat(),
                "bias": article.bias_label,
                "source": article.source,
                "title": article.title,
                "topic": browseable.topic_cluster
            })
        
        # Get statistics
        stats = browser.get_statistics(browseable_articles)
        
        return JSONResponse({
            "articles": formatted_articles,
            "temporal_data": temporal_data,
            "stats": {
                "total_articles": stats["total_articles"],
                "articles_with_perspectives": stats["articles_with_perspectives"],
                "perspective_coverage": stats["perspective_coverage"],
                "average_perspectives": stats["average_perspectives_per_article"],
                "max_perspectives": stats["max_perspectives_found"],
                "bias_distribution": stats["bias_distribution"]
            },
            "query": query,
            "days": days,
            "threshold": threshold
        })
        
    except Exception as e:
        print(f"❌ API error: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse({
            "error": f"Failed to process request: {str(e)}",
            "articles": [],
            "temporal_data": []
        })

def get_bias_name(bias_label):
    """Convert bias label to readable name"""
    bias_names = {0: "Left", 1: "Center", 2: "Right", None: "Unknown"}
    return bias_names.get(bias_label, "Unknown")

def get_frontend_html():
    """Return the complete frontend HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📰 News Perspective Explorer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .search-panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .search-row {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .search-input {
            flex: 1;
            padding: 12px 20px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            min-width: 200px;
        }
        
        .search-input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        select, input[type="number"] {
            padding: 12px 15px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 14px;
            background: white;
        }
        
        .search-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .search-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .search-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
        
        .stats-panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .stat-item:last-child {
            border-bottom: none;
        }
        
        .stat-label {
            font-weight: 600;
            color: #555;
        }
        
        .stat-value {
            font-weight: 700;
            font-size: 18px;
            color: #667eea;
        }
        
        .chart-panel {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .chart-title {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 20px;
            color: #333;
            text-align: center;
        }
        
        .articles-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        
        .article-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .article-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        
        .article-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 15px;
        }
        
        .bias-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .bias-left {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .bias-center {
            background: #f3e5f5;
            color: #7b1fa2;
        }
        
        .bias-right {
            background: #ffebee;
            color: #d32f2f;
        }
        
        .article-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 10px;
            line-height: 1.4;
            color: #333;
        }
        
        .article-title a {
            color: inherit;
            text-decoration: none;
        }
        
        .article-title a:hover {
            color: #667eea;
        }
        
        .article-meta {
            color: #666;
            font-size: 14px;
            margin-bottom: 15px;
        }
        
        .perspectives-section {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #f0f0f0;
        }
        
        .perspectives-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .perspective-item {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid transparent;
        }
        
        .perspective-item.left {
            border-left-color: #1976d2;
        }
        
        .perspective-item.center {
            border-left-color: #7b1fa2;
        }
        
        .perspective-item.right {
            border-left-color: #d32f2f;
        }
        
        .perspective-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .perspective-title a {
            color: #333;
            text-decoration: none;
        }
        
        .perspective-title a:hover {
            color: #667eea;
        }
        
        .perspective-meta {
            font-size: 12px;
            color: #666;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .similarity-score {
            background: #667eea;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            color: white;
            font-size: 18px;
        }
        
        .error {
            background: #ffebee;
            color: #d32f2f;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
        
        .no-results {
            text-align: center;
            padding: 50px;
            color: #666;
            font-size: 18px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📰 News Perspective Explorer</h1>
            <p>Discover diverse viewpoints on the same stories</p>
        </div>
        
        <div class="search-panel">
            <div class="search-row">
                <input type="text" id="searchQuery" class="search-input" placeholder="Search for news topics (e.g., election, climate, economy)" value="">
                
                <select id="daysBack">
                    <option value="3">Last 3 days</option>
                    <option value="7" selected>Last 7 days</option>
                    <option value="14">Last 14 days</option>
                    <option value="21">Last 21 days</option>
                </select>
                
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label for="threshold" style="font-size: 14px; color: #666;">Precision:</label>
                    <input type="number" id="threshold" min="0.5" max="0.9" step="0.05" value="0.70" style="width: 80px;">
                </div>
                
                <button class="search-btn" onclick="searchArticles()">
                    🔍 Explore Perspectives
                </button>
            </div>
        </div>
        
        <div id="results"></div>
    </div>

    <script>
        let currentData = null;
        let temporalChart = null;
        
        // Search function
        async function searchArticles() {
            const query = document.getElementById('searchQuery').value;
            const days = document.getElementById('daysBack').value;
            const threshold = document.getElementById('threshold').value;
            const button = document.querySelector('.search-btn');
            
            // Show loading
            button.disabled = true;
            button.innerHTML = '🔄 Exploring...';
            document.getElementById('results').innerHTML = '<div class="loading pulse">🔍 Finding diverse perspectives...</div>';
            
            try {
                const response = await fetch(`/api/articles?query=${encodeURIComponent(query)}&days=${days}&threshold=${threshold}`);
                const data = await response.json();
                
                currentData = data;
                displayResults(data);
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('results').innerHTML = `
                    <div class="error">
                        ❌ Error loading articles: ${error.message}
                        <br><br>
                        💡 Try checking your internet connection or refreshing the page.
                    </div>
                `;
            } finally {
                button.disabled = false;
                button.innerHTML = '🔍 Explore Perspectives';
            }
        }
        
        function displayResults(data) {
            if (data.error) {
                document.getElementById('results').innerHTML = `
                    <div class="error">
                        ❌ ${data.error}
                        <br><br>
                        💡 Try checking your API keys or network connection.
                    </div>
                `;
                return;
            }
            
            if (!data.articles || data.articles.length === 0) {
                document.getElementById('results').innerHTML = `
                    <div class="no-results">
                        📭 No articles found
                        <br><br>
                        💡 Try:
                        <br>• A broader search query
                        <br>• More days in the time range
                        <br>• Lower precision threshold
                    </div>
                `;
                return;
            }
            
            const html = `
                <div class="dashboard">
                    <div class="chart-panel">
                        <div class="chart-title">📊 Bias Timeline</div>
                        <canvas id="temporalChart" width="400" height="200"></canvas>
                    </div>
                    
                    <div class="stats-panel">
                        <h3 style="margin-bottom: 20px; color: #333;">📈 Analysis Summary</h3>
                        <div class="stat-item">
                            <span class="stat-label">Total Articles</span>
                            <span class="stat-value">${data.stats.total_articles}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">With Perspectives</span>
                            <span class="stat-value">${data.stats.articles_with_perspectives}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Coverage</span>
                            <span class="stat-value">${(data.stats.perspective_coverage * 100).toFixed(1)}%</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Avg Perspectives</span>
                            <span class="stat-value">${data.stats.average_perspectives.toFixed(1)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Max Found</span>
                            <span class="stat-value">${data.stats.max_perspectives}</span>
                        </div>
                        
                        <h4 style="margin: 20px 0 10px 0; color: #333;">🏛️ Source Distribution</h4>
                        ${Object.entries(data.stats.bias_distribution).map(([bias, count]) => `
                            <div class="stat-item">
                                <span class="stat-label">${bias}</span>
                                <span class="stat-value">${count}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                <div class="articles-grid">
                    ${data.articles.filter(article => article.perspective_count > 0).map(article => `
                        <div class="article-card">
                            <div class="article-header">
                                <span class="bias-badge bias-${article.bias_name.toLowerCase()}">${article.bias_name}</span>
                                <span style="font-size: 14px; color: #666;">${formatDate(article.published_at)}</span>
                            </div>
                            
                            <h3 class="article-title">
                                <a href="${article.url}" target="_blank">${article.title}</a>
                            </h3>
                            
                            <div class="article-meta">
                                📰 ${article.source} | 🏷️ ${article.topic}
                            </div>
                            
                            ${article.description ? `<p style="color: #666; margin-bottom: 15px; line-height: 1.5;">${article.description}</p>` : ''}
                            
                            ${article.perspectives.length > 0 ? `
                                <div class="perspectives-section">
                                    <div class="perspectives-title">
                                        🎯 Alternative Perspectives (${article.perspectives.length})
                                    </div>
                                    ${article.perspectives.map(perspective => `
                                        <div class="perspective-item ${perspective.bias_name.toLowerCase()}">
                                            <div class="perspective-title">
                                                <a href="${perspective.url}" target="_blank">${perspective.title}</a>
                                            </div>
                                            <div class="perspective-meta">
                                                <span>📰 ${perspective.source}</span>
                                                <span class="similarity-score">${(perspective.similarity * 100).toFixed(1)}% match</span>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>
            `;
            
            document.getElementById('results').innerHTML = html;
            
            // Create temporal chart
            createTemporalChart(data.temporal_data);
        }
        
        function createTemporalChart(temporalData) {
            const ctx = document.getElementById('temporalChart');
            if (!ctx) return;
            
            // Process temporal data
            const chartData = processTemporalData(temporalData);
            
            if (temporalChart) {
                temporalChart.destroy();
            }
            
            temporalChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.labels,
                    datasets: [
                        {
                            label: 'Left-leaning',
                            data: chartData.leftData,
                            borderColor: '#1976d2',
                            backgroundColor: 'rgba(25, 118, 210, 0.1)',
                            tension: 0.4,
                            fill: false
                        },
                        {
                            label: 'Center',
                            data: chartData.centerData,
                            borderColor: '#7b1fa2',
                            backgroundColor: 'rgba(123, 31, 162, 0.1)',
                            tension: 0.4,
                            fill: false
                        },
                        {
                            label: 'Right-leaning',
                            data: chartData.rightData,
                            borderColor: '#d32f2f',
                            backgroundColor: 'rgba(211, 47, 47, 0.1)',
                            tension: 0.4,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Number of Articles'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Political Bias Over Time'
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    }
                }
            });
        }
        
        function processTemporalData(temporalData) {
            // Group by date
            const dateGroups = {};
            
            temporalData.forEach(item => {
                const date = new Date(item.date).toLocaleDateString();
                if (!dateGroups[date]) {
                    dateGroups[date] = { left: 0, center: 0, right: 0 };
                }
                
                if (item.bias === 0) dateGroups[date].left++;
                else if (item.bias === 1) dateGroups[date].center++;
                else if (item.bias === 2) dateGroups[date].right++;
            });
            
            // Sort dates
            const sortedDates = Object.keys(dateGroups).sort((a, b) => new Date(a) - new Date(b));
            
            return {
                labels: sortedDates,
                leftData: sortedDates.map(date => dateGroups[date].left),
                centerData: sortedDates.map(date => dateGroups[date].center),
                rightData: sortedDates.map(date => dateGroups[date].right)
            };
        }
        
        function formatDate(dateString) {
            const date = new Date(dateString);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        }
        
        // Search on Enter key
        document.getElementById('searchQuery').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchArticles();
            }
        });
        
        // Load initial results
        window.addEventListener('load', function() {
            searchArticles();
        });
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    print("🚀 Starting Simple News Perspective Web Server")
    print("=" * 50)
    print("🌐 Server will be available at: http://localhost:8000")
    print("📊 Features:")
    print("   • Interactive temporal bias chart")
    print("   • Adjustable precision threshold")
    print("   • Real-time perspective discovery")
    print("   • Simple vanilla JavaScript frontend")
    print("")
    
    uvicorn.run(
        "simple_web_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Disable reload to avoid import issues
        log_level="info"
    )