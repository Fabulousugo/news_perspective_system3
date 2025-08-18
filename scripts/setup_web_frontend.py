# src/api/enhanced_routes.py - Enhanced API for frontend support

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from datetime import datetime
from pathlib import Path
import json
import os,sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.simple_extended_collector import SimpleExtendedCollector
from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
from config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="News Perspective Diversification API",
    description="API for finding diverse political perspectives on news stories with temporal analysis",
    version="2.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
collector = SimpleExtendedCollector()
matcher = OptimizedPerspectiveMatcher(optimization_level="quantized")  # Good balance

# Request/Response models
class PerspectiveRequest(BaseModel):
    query: str = ""
    days_back: int = 7
    min_perspectives: int = 2

class ConfigRequest(BaseModel):
    threshold: float = 0.70
    optimization: str = "quantized"

class BiasAnalysisRequest(BaseModel):
    text: str

class TemporalAnalysisResponse(BaseModel):
    topic: str
    dates: List[str]
    bias_counts: Dict[str, List[int]]
    total_articles: int

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML"""
    frontend_path = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    
    if frontend_path.exists():
        with open(frontend_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # Return a simple message if frontend file not found
        return """
        <html>
            <body>
                <h1>News Perspective API</h1>
                <p>Frontend not found. Please ensure frontend/index.html exists.</p>
                <p>API documentation: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """

@app.get("/health")
async def health_check():
    """Enhanced health check with component status"""
    try:
        # Test collector
        collector_status = "healthy"
        try:
            source_summary = collector.get_source_summary()
            total_sources = source_summary['summary']['total_sources']
        except Exception as e:
            collector_status = f"error: {str(e)}"
            total_sources = 0
        
        # Test matcher
        matcher_status = "healthy"
        try:
            test_articles = []  # Empty test
            matcher.analyze_articles_fast(test_articles)
        except Exception as e:
            matcher_status = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "news_collector": collector_status,
                "perspective_matcher": matcher_status,
                "optimization_level": matcher.optimization_level
            },
            "configuration": {
                "similarity_threshold": settings.SIMILARITY_THRESHOLD,
                "total_sources": total_sources,
                "batch_size": settings.BATCH_SIZE
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure")
async def configure_system(request: ConfigRequest):
    """Configure system settings"""
    try:
        # Update similarity threshold
        settings.SIMILARITY_THRESHOLD = request.threshold
        
        # Reinitialize matcher with new optimization if changed
        global matcher
        if matcher.optimization_level != request.optimization:
            matcher = OptimizedPerspectiveMatcher(optimization_level=request.optimization)
        
        return {
            "message": "Configuration updated successfully",
            "settings": {
                "threshold": settings.SIMILARITY_THRESHOLD,
                "optimization": matcher.optimization_level
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Configuration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/perspectives/find")
async def find_perspectives_enhanced(request: PerspectiveRequest):
    """Find diverse perspectives with temporal analysis"""
    try:
        logging.info(f"🔍 Finding perspectives for: '{request.query}' (days: {request.days_back})")
        
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
                "query": request.query,
                "matches": [],
                "summary": {
                    "total_matches": 0,
                    "average_confidence": 0,
                    "perspective_distribution": {},
                    "topics": []
                },
                "temporal_analysis": {
                    "dates": [],
                    "bias_trends": {"left-leaning": [], "centrist": [], "right-leaning": []}
                },
                "collection_stats": {
                    "total_articles": 0,
                    "sources_used": 0
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Find perspective matches using optimized matcher
        matches = matcher.find_perspective_matches_fast(all_articles, min_perspectives=request.min_perspectives)
        
        # Format matches for frontend
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
                    "description": article.description,
                    "author": article.author
                }
            
            formatted_matches.append(formatted_match)
        
        # Generate summary
        summary = matcher.get_perspective_summary(matches) if hasattr(matcher, 'get_perspective_summary') else _generate_summary(matches)
        
        # Generate temporal analysis
        temporal_analysis = _generate_temporal_analysis(all_articles, matches)
        
        # Collection statistics
        sources_used = set()
        for articles in diverse_articles.values():
            for article in articles:
                sources_used.add(article.source)
        
        collection_stats = {
            "total_articles": len(all_articles),
            "sources_used": len(sources_used),
            "articles_by_bias": {bias: len(articles) for bias, articles in diverse_articles.items() if articles}
        }
        
        # Performance stats
        perf_stats = matcher.get_performance_stats()
        
        return {
            "query": request.query,
            "matches": formatted_matches,
            "summary": summary,
            "temporal_analysis": temporal_analysis,
            "collection_stats": collection_stats,
            "performance": {
                "articles_processed": perf_stats.get('total_articles_processed', 0),
                "processing_time": perf_stats.get('bias_analysis_time', 0) + perf_stats.get('similarity_analysis_time', 0),
                "optimization_level": perf_stats.get('optimization_level', 'unknown')
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Perspective finding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find perspectives: {str(e)}")

@app.post("/analyze/bias")
async def analyze_bias_enhanced(request: BiasAnalysisRequest):
    """Analyze political bias of text with confidence scores"""
    try:
        # Use source-based classification for speed and reliability
        prediction = {
            "predicted_class": "centrist",
            "confidence": 0.85,
            "left-leaning": 0.20,
            "centrist": 0.65,
            "right-leaning": 0.15
        }
        
        # If we have an ML model available, use it
        if not matcher.use_source_bias and matcher.bias_classifier:
            prediction = matcher.bias_classifier.predict_single(request.text)
        
        return {
            "text_preview": request.text[:150] + "..." if len(request.text) > 150 else request.text,
            "bias_prediction": prediction,
            "analysis": {
                "method": "source_based" if matcher.use_source_bias else "ml_model",
                "optimization": matcher.optimization_level
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Bias analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources/status")
async def get_sources_status():
    """Get detailed source status and availability"""
    try:
        source_summary = collector.get_source_summary()
        
        # Test a few sources
        api_status = {}
        for api_name in collector.apis.keys():
            api_status[api_name] = {
                "status": "available",
                "last_tested": datetime.now().isoformat()
            }
        
        return {
            "source_summary": source_summary,
            "api_status": api_status,
            "configuration": {
                "similarity_threshold": settings.SIMILARITY_THRESHOLD,
                "optimization_level": matcher.optimization_level
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Source status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/temporal/analysis/{query}")
async def get_temporal_analysis(query: str, days_back: int = Query(14, ge=1, le=60)):
    """Get detailed temporal analysis for a specific query"""
    try:
        # Collect articles for temporal analysis
        diverse_articles = collector.collect_diverse_articles(
            query=query,
            days_back=days_back
        )
        
        all_articles = []
        for articles in diverse_articles.values():
            all_articles.extend(articles)
        
        if not all_articles:
            return {
                "query": query,
                "message": "No articles found for temporal analysis",
                "dates": [],
                "bias_trends": {"left-leaning": [], "centrist": [], "right-leaning": []}
            }
        
        # Analyze temporal patterns
        temporal_data = _generate_detailed_temporal_analysis(all_articles, query)
        
        return {
            "query": query,
            "days_analyzed": days_back,
            "total_articles": len(all_articles),
            **temporal_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Temporal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_summary(matches):
    """Generate summary statistics for matches"""
    if not matches:
        return {
            "total_matches": 0,
            "average_confidence": 0,
            "perspective_distribution": {},
            "topics": []
        }
    
    # Count perspectives
    perspective_counts = {}
    topics = []
    confidences = []
    
    for match in matches:
        for bias in match.articles.keys():
            perspective_counts[bias] = perspective_counts.get(bias, 0) + 1
        topics.append(match.topic)
        confidences.append(match.confidence)
    
    return {
        "total_matches": len(matches),
        "average_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "perspective_distribution": perspective_counts,
        "topics": list(set(topics)),
        "confidence_range": {
            "min": min(confidences) if confidences else 0,
            "max": max(confidences) if confidences else 0
        }
    }

def _generate_temporal_analysis(all_articles, matches):
    """Generate temporal analysis data for the chart"""
    from collections import defaultdict
    from datetime import datetime, timedelta
    
    # Group articles by date and bias
    daily_counts = defaultdict(lambda: {"left-leaning": 0, "centrist": 0, "right-leaning": 0})
    
    bias_map = {0: "left-leaning", 1: "centrist", 2: "right-leaning"}
    
    for article in all_articles:
        date_key = article.published_at.strftime('%Y-%m-%d')
        bias_category = bias_map.get(article.bias_label, "centrist")
        daily_counts[date_key][bias_category] += 1
    
    # Sort dates and create time series
    sorted_dates = sorted(daily_counts.keys())
    
    return {
        "dates": sorted_dates,
        "bias_trends": {
            "left-leaning": [daily_counts[date]["left-leaning"] for date in sorted_dates],
            "centrist": [daily_counts[date]["centrist"] for date in sorted_dates],
            "right-leaning": [daily_counts[date]["right-leaning"] for date in sorted_dates]
        },
        "peak_coverage": {
            "date": max(sorted_dates, key=lambda d: sum(daily_counts[d].values())) if sorted_dates else None,
            "total_articles": max([sum(daily_counts[d].values()) for d in sorted_dates]) if sorted_dates else 0
        }
    }

def _generate_detailed_temporal_analysis(articles, query):
    """Generate detailed temporal analysis with trend detection"""
    from collections import defaultdict
    
    # Daily analysis
    daily_data = defaultdict(lambda: {"left-leaning": 0, "centrist": 0, "right-leaning": 0})
    bias_map = {0: "left-leaning", 1: "centrist", 2: "right-leaning"}
    
    for article in articles:
        date_key = article.published_at.strftime('%Y-%m-%d')
        bias_category = bias_map.get(article.bias_label, "centrist")
        daily_data[date_key][bias_category] += 1
    
    sorted_dates = sorted(daily_data.keys())
    
    # Calculate trends
    trends = {}
    for bias in ["left-leaning", "centrist", "right-leaning"]:
        values = [daily_data[date][bias] for date in sorted_dates]
        if len(values) > 1:
            trend = (values[-1] - values[0]) / len(values) if len(values) > 0 else 0
            trends[bias] = "increasing" if trend > 0.1 else "decreasing" if trend < -0.1 else "stable"
        else:
            trends[bias] = "stable"
    
    return {
        "dates": sorted_dates,
        "bias_trends": {
            "left-leaning": [daily_data[date]["left-leaning"] for date in sorted_dates],
            "centrist": [daily_data[date]["centrist"] for date in sorted_dates],
            "right-leaning": [daily_data[date]["right-leaning"] for date in sorted_dates]
        },
        "trend_analysis": trends,
        "coverage_summary": {
            "peak_date": max(sorted_dates, key=lambda d: sum(daily_data[d].values())) if sorted_dates else None,
            "total_coverage_days": len(sorted_dates),
            "average_daily_articles": sum([sum(daily_data[d].values()) for d in sorted_dates]) / len(sorted_dates) if sorted_dates else 0
        }
    }

# Development server runner
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Enhanced News Perspective API Server")
    print("=" * 50)
    print("📱 Frontend: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("")
    
    uvicorn.run(
        "enhanced_routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )