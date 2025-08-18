# src/api/simple_routes.py - Simple, working API without import errors

import sys, os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Simple imports to avoid ASGI errors
try:
    from src.data_collection.simple_extended_collector import SimpleExtendedCollector
    from src.models.news_browser import NewsBrowser
    from config.settings import settings
except ImportError:
    # Fallback imports
    import sys
    sys.path.append("src")
    from data_collection.simple_extended_collector import SimpleExtendedCollector
    from models.news_browser import NewsBrowser
    from config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="News Perspective API",
    description="Simple API for news perspective analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for frontend)
try:
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
except:
    pass  # Directory might not exist yet

# Global components (initialize once)
collector = SimpleExtendedCollector()
browser = NewsBrowser()

# Request/Response models
class SearchRequest(BaseModel):
    query: str = ""
    days: int = 7
    threshold: float = 0.65
    limit: int = 20

class ThresholdUpdate(BaseModel):
    threshold: float

# Global threshold setting
current_threshold = 0.65

@app.get("/")
async def root():
    """Serve the main frontend page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>News Perspective Browser</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
        <h1>🚀 News Perspective Browser</h1>
        <p>API is running! Frontend will be available soon.</p>
        <div>
            <h3>📡 API Endpoints:</h3>
            <ul>
                <li><a href="/docs">/docs</a> - API Documentation</li>
                <li><a href="/api/health">/api/health</a> - Health Check</li>
                <li><a href="/api/sources">/api/sources</a> - Available Sources</li>
            </ul>
        </div>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </body>
    </html>
    """)

@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "threshold": current_threshold,
        "components": {
            "collector": "active",
            "browser": "active"
        }
    }

@app.get("/api/sources")
async def get_sources():
    """Get available news sources"""
    try:
        summary = collector.get_source_summary()
        return {
            "status": "success",
            "sources": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Sources endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/threshold")
async def update_threshold(request: ThresholdUpdate):
    """Update similarity threshold"""
    global current_threshold
    
    if 0.1 <= request.threshold <= 1.0:
        current_threshold = request.threshold
        settings.SIMILARITY_THRESHOLD = request.threshold
        browser.similarity_detector.threshold = request.threshold
        
        return {
            "status": "success",
            "new_threshold": current_threshold,
            "message": f"Threshold updated to {current_threshold}"
        }
    else:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.1 and 1.0")

@app.post("/api/search")
async def search_articles(request: SearchRequest):
    """Search for articles with perspective analysis"""
    try:
        start_time = time.time()
        
        # Update threshold if provided
        if request.threshold != current_threshold:
            settings.SIMILARITY_THRESHOLD = request.threshold
            browser.similarity_detector.threshold = request.threshold
        
        # Collect articles
        logging.info(f"Searching for '{request.query}' with threshold {request.threshold}")
        
        diverse_articles = collector.collect_diverse_articles(
            query=request.query,
            days_back=request.days
        )
        
        # Flatten articles
        all_articles = []
        for bias_category, articles in diverse_articles.items():
            all_articles.extend(articles)
        
        if not all_articles:
            return {
                "status": "no_results",
                "message": "No articles found",
                "search_params": {
                    "query": request.query,
                    "days": request.days,
                    "threshold": request.threshold
                }
            }
        
        # Analyze for perspectives
        browseable_articles = browser.browse_articles(all_articles)
        stats = browser.get_statistics(browseable_articles)
        
        # Format articles for frontend
        formatted_articles = []
        for i, browseable in enumerate(browseable_articles[:request.limit]):
            article = browseable.article
            
            # Format perspectives
            perspectives = []
            for related_article, similarity in browseable.related_articles:
                perspectives.append({
                    "title": related_article.title,
                    "source": related_article.source,
                    "url": related_article.url,
                    "bias_label": related_article.bias_label,
                    "bias_name": {0: "left", 1: "center", 2: "right"}.get(related_article.bias_label, "unknown"),
                    "similarity": similarity,
                    "published_at": related_article.published_at.isoformat()
                })
            
            formatted_articles.append({
                "id": i,
                "title": article.title,
                "source": article.source,
                "url": article.url,
                "description": article.description,
                "published_at": article.published_at.isoformat(),
                "bias_label": article.bias_label,
                "bias_name": {0: "left", 1: "center", 2: "right"}.get(article.bias_label, "unknown"),
                "perspective_count": browseable.perspective_count,
                "perspectives": perspectives,
                "topic": browseable.topic_cluster
            })
        
        # Create temporal data for bias chart
        temporal_data = create_temporal_bias_data(all_articles)
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "query": request.query,
            "articles": formatted_articles,
            "statistics": {
                "total_articles": stats['total_articles'],
                "articles_with_perspectives": stats['articles_with_perspectives'],
                "perspective_coverage": stats['perspective_coverage'],
                "average_perspectives": stats['average_perspectives_per_article'],
                "max_perspectives": stats['max_perspectives_found'],
                "bias_distribution": stats['bias_distribution']
            },
            "temporal_bias": temporal_data,
            "processing_time": processing_time,
            "search_params": {
                "query": request.query,
                "days": request.days,
                "threshold": request.threshold,
                "limit": request.limit
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/article/{article_id}")
async def get_article_details(article_id: int):
    """Get detailed view of a specific article"""
    # This would be implemented with article storage
    # For now, return a placeholder
    return {
        "status": "success",
        "article_id": article_id,
        "message": "Article details endpoint - to be implemented"
    }

def create_temporal_bias_data(articles: List) -> List[Dict]:
    """Create temporal bias data for charting"""
    if not articles:
        return []
    
    # Group articles by day and bias
    temporal_data = {}
    bias_names = {0: "left", 1: "center", 2: "right"}
    
    for article in articles:
        # Get date string (YYYY-MM-DD)
        date_str = article.published_at.strftime("%Y-%m-%d")
        
        if date_str not in temporal_data:
            temporal_data[date_str] = {"left": 0, "center": 0, "right": 0, "date": date_str}
        
        bias_name = bias_names.get(article.bias_label, "center")
        temporal_data[date_str][bias_name] += 1
    
    # Convert to list and sort by date
    chart_data = list(temporal_data.values())
    chart_data.sort(key=lambda x: x["date"])
    
    return chart_data

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return HTMLResponse(
        content="<h1>404 - Not Found</h1><p>The requested resource was not found.</p>",
        status_code=404
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "status": "error",
        "message": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logging.info("🚀 News Perspective API starting up...")
    logging.info(f"📊 Current threshold: {current_threshold}")
    
    # Test components
    try:
        summary = collector.get_source_summary()
        logging.info(f"✅ Collector ready with {summary['summary']['total_sources']} sources")
    except Exception as e:
        logging.error(f"❌ Collector initialization failed: {e}")
    
    logging.info("✅ API ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")