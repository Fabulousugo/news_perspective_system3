#!/usr/bin/env python3
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
