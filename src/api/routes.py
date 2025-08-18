# src/api/routes.py

# src/api/routes.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime

from ..data_collection.news_apis import NewsCollector
from ..models.perspective_matcher import PerspectiveMatcher
from ..models.bias_classifier import BiasClassifier
from ..models.similarity_detector import SimilarityDetector

# Initialize FastAPI app
app = FastAPI(
    title="News Perspective Diversification API",
    description="API for finding diverse political perspectives on news stories",
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

# Initialize components
collector = NewsCollector()
matcher = PerspectiveMatcher()

# Request/Response models
class ArticleRequest(BaseModel):
    query: str = ""
    days_back: int = 7
    min_perspectives: int = 2

class BiasAnalysisRequest(BaseModel):
    text: str

class SimilarityRequest(BaseModel):
    query_text: str
    candidate_texts: List[str]
    top_k: int = 5

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "News Perspective Diversification API", 
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "news_collector": "active",
            "bias_classifier": "active", 
            "similarity_detector": "active",
            "perspective_matcher": "active"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze/bias")
async def analyze_bias(request: BiasAnalysisRequest):
    """Analyze political bias of a text"""
    try:
        prediction = matcher.bias_classifier.predict_single(request.text)
        return {
            "text_preview": request.text[:100] + "..." if len(request.text) > 100 else request.text,
            "bias_prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Bias analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/similarity")
async def analyze_similarity(request: SimilarityRequest):
    """Find similar articles to a query"""
    try:
        similar_articles = matcher.similarity_detector.find_similar_articles(
            request.query_text,
            request.candidate_texts,
            top_k=request.top_k
        )
        
        return {
            "query_preview": request.query_text[:100] + "..." if len(request.query_text) > 100 else request.query_text,
            "similar_articles": [
                {
                    "index": idx,
                    "similarity_score": score,
                    "text_preview": request.candidate_texts[idx][:100] + "..."
                }
                for idx, score in similar_articles
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Similarity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/articles/collect")
async def collect_articles(request: ArticleRequest):
    """Collect articles from diverse sources"""
    try:
        # Collect articles
        diverse_articles = collector.collect_diverse_articles(
            query=request.query,
            days_back=request.days_back
        )
        
        # Format response
        response = {
            "query": request.query,
            "days_back": request.days_back,
            "collection_summary": {},
            "articles_by_bias": {}
        }
        
        total_articles = 0
        for bias_category, articles in diverse_articles.items():
            article_count = len(articles)
            total_articles += article_count
            
            response["collection_summary"][bias_category] = article_count
            response["articles_by_bias"][bias_category] = [
                {
                    "title": article.title,
                    "source": article.source,
                    "url": article.url,
                    "published_at": article.published_at.isoformat(),
                    "description": article.description,
                    "bias_label": article.bias_label
                }
                for article in articles[:10]  # Limit to first 10 per category
            ]
        
        response["total_articles"] = total_articles
        response["timestamp"] = datetime.now().isoformat()
        
        return response
        
    except Exception as e:
        logging.error(f"Article collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/perspectives/find")
async def find_perspectives(request: ArticleRequest):
    """Find articles covering same stories from different perspectives"""
    try:
        # Collect articles
        diverse_articles = collector.collect_diverse_articles(
            query=request.query,
            days_back=request.days_back
        )
        
        # Flatten articles
        all_articles = []
        for bias_category, articles in diverse_articles.items():
            all_articles.extend(articles)
        
        if not all_articles:
            return {
                "message": "No articles found for the given query",
                "matches": [],
                "summary": {}
            }
        
        # Find perspective matches
        matches = matcher.find_perspective_matches(
            all_articles,
            min_perspectives=request.min_perspectives
        )
        
        # Format response
        formatted_matches = []
        for match in matches:
            formatted_match = {
                "story_id": match.story_id,
                "topic": match.topic,
                "confidence": match.confidence,
                "perspectives": {}
            }
            
            for bias, article in match.articles.items():
                formatted_match["perspectives"][bias] = {
                    "title": article.title,
                    "source": article.source,
                    "url": article.url,
                    "published_at": article.published_at.isoformat(),
                    "description": article.description
                }
            
            formatted_matches.append(formatted_match)
        
        # Get summary
        summary = matcher.get_perspective_summary(matches)
        
        return {
            "query": request.query,
            "matches": formatted_matches,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Perspective finding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources/status")
async def get_source_status():
    """Get status of configured news sources"""
    try:
        sources_config = collector.source_configs
        
        # Test API availability (simplified)
        api_status = {}
        for api_name in collector.apis.keys():
            api_status[api_name] = "available"
        
        return {
            "configured_sources": sources_config['news_sources'],
            "api_status": api_status,
            "total_sources": sum(
                len(sources) for sources in sources_config['news_sources'].values()
            ),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Source status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/components/status")
async def get_component_status():
    """Get status of configured components"""
    return {
        "components": {
            "news_collector": "active",
            "bias_classifier": "active", 
            "similarity_detector": "active",
            "perspective_matcher": "active"
        },
        "timestamp": datetime.now().isoformat()
    }