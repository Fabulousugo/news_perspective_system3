# src/api/fast_routes.py - Optimized API routes for the fast frontend

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime
import json
import asyncio
from pathlib import Path

# Try to import the working components
try:
    from ..data_collection.news_apis import NewsCollector
    from ..models.similarity_detector import SimilarityDetector
except ImportError:
    # Fallback imports
    import sys
    sys.path.append("src")
    from data_collection.news_apis import NewsCollector
    from models.similarity_detector import SimilarityDetector

app = FastAPI(
    title="Fast News Perspective API",
    description="Optimized API for fast news perspective discovery",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (lazy loading)
collector = None
similarity_detector = None

def get_collector():
    global collector
    if collector is None:
        collector = NewsCollector()
    return collector

def get_similarity_detector():
    global similarity_detector
    if similarity_detector is None:
        similarity_detector = SimilarityDetector()
    return similarity_detector

# Request/Response models
class ArticleCollectionRequest(BaseModel):
    query: str = ""
    days_back: int = 7

class SimilarityRequest(BaseModel):
    query_text: str
    candidate_texts: List[str]
    top_k: int = 5

class FastStatsResponse(BaseModel):
    total_articles: int
    articles_by_bias: Dict[str, int]
    collection_time: str
    timestamp: str

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the fast frontend HTML"""
    frontend_path = Path(__file__).parent.parent.parent / "fast_news_frontend.html"
    
    if frontend_path.exists():
        with open(frontend_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>News Perspective Explorer</h1>
                <p>Frontend file not found. Please ensure fast_news_frontend.html is in the project root.</p>
                <p>Available endpoints:</p>
                <ul>
                    <li><a href="/docs">API Documentation</a></li>
                    <li><a href="/articles/collect">Articles Collection (POST)</a></li>
                    <li><a href="/analyze/similarity">Similarity Analysis (POST)</a></li>
                </ul>
            </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Quick health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/articles/collect")
async def collect_articles_fast(request: ArticleCollectionRequest):
    """
    Fast article collection endpoint - optimized for speed
    Returns articles immediately without perspective matching
    """
    try:
        start_time = datetime.now()
        
        # Get collector and fetch articles
        news_collector = get_collector()
        
        logging.info(f"Fast collection: query='{request.query}', days={request.days_back}")
        
        # Collect articles from diverse sources
        diverse_articles = news_collector.collect_diverse_articles(
            query=request.query,
            days_back=request.days_back
        )
        
        collection_time = (datetime.now() - start_time).total_seconds()
        
        # Format response for frontend
        total_articles = sum(len(articles) for articles in diverse_articles.values())
        
        # Convert articles to JSON-serializable format
        formatted_articles = {}
        
        for bias_category, articles in diverse_articles.items():
            formatted_articles[bias_category] = []
            
            for article in articles:
                formatted_article = {
                    "title": article.title,
                    "description": article.description,
                    "url": article.url,
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "author": article.author,
                    "bias_label": article.bias_label,
                    "content_preview": article.content[:200] + "..." if len(article.content) > 200 else article.content
                }
                formatted_articles[bias_category].append(formatted_article)
        
        # Create summary stats
        collection_summary = {bias: len(articles) for bias, articles in diverse_articles.items()}
        
        response = {
            "success": True,
            "query": request.query,
            "days_back": request.days_back,
            "total_articles": total_articles,
            "collection_time_seconds": round(collection_time, 2),
            "collection_summary": collection_summary,
            "articles_by_bias": formatted_articles,
            "timestamp": datetime.now().isoformat(),
            "message": f"Collected {total_articles} articles in {collection_time:.2f}s"
        }
        
        logging.info(f"Fast collection complete: {total_articles} articles in {collection_time:.2f}s")
        return response
        
    except Exception as e:
        logging.error(f"Article collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to collect articles: {str(e)}")

@app.post("/analyze/similarity")
async def analyze_similarity_fast(request: SimilarityRequest):
    """
    Fast similarity analysis endpoint
    Optimized for real-time perspective matching
    """
    try:
        start_time = datetime.now()
        
        # Get similarity detector
        detector = get_similarity_detector()
        
        # Find similar articles
        similar_articles = detector.find_similar_articles(
            request.query_text,
            request.candidate_texts,
            top_k=request.top_k
        )
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        # Format response
        response = {
            "success": True,
            "query_preview": request.query_text[:100] + "..." if len(request.query_text) > 100 else request.query_text,
            "candidates_count": len(request.candidate_texts),
            "similar_articles": [
                {
                    "index": idx,
                    "similarity_score": score,
                    "text_preview": request.candidate_texts[idx][:100] + "..." if len(request.candidate_texts[idx]) > 100 else request.candidate_texts[idx]
                }
                for idx, score in similar_articles
            ],
            "analysis_time_seconds": round(analysis_time, 3),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logging.error(f"Similarity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity analysis failed: {str(e)}")

@app.get("/stats/quick")
async def get_quick_stats():
    """Get quick system statistics"""
    try:
        # This could be cached for better performance
        return {
            "system_status": "operational",
            "available_endpoints": [
                "/articles/collect",
                "/analyze/similarity", 
                "/stats/quick"
            ],
            "features": [
                "Fast article collection",
                "Progressive perspective loading",
                "Real-time similarity analysis",
                "Multi-source aggregation"
            ],
            "sources": {
                "api_sources": ["NewsAPI", "Guardian"],
                "bias_categories": ["left-leaning", "centrist", "right-leaning"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources/status")
async def get_sources_status():
    """Get status of news sources"""
    try:
        news_collector = get_collector()
        
        # Quick source check
        available_apis = list(news_collector.apis.keys())
        
        return {
            "available_apis": available_apis,
            "api_count": len(available_apis),
            "status": "operational" if available_apis else "limited",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return {
        "error": "Endpoint not found",
        "available_endpoints": [
            "/",
            "/docs", 
            "/articles/collect",
            "/analyze/similarity",
            "/stats/quick",
            "/sources/status"
        ]
    }

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return {
        "error": "Internal server error",
        "message": "Please check the server logs for details",
        "timestamp": datetime.now().isoformat()
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logging.info("🚀 Fast News Perspective API starting up...")
    logging.info("📰 Optimized for progressive loading and real-time interaction")
    
    # Pre-initialize components for faster first request
    try:
        get_collector()
        get_similarity_detector()
        logging.info("✅ Components pre-loaded successfully")
    except Exception as e:
        logging.warning(f"⚠️  Component pre-loading failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)