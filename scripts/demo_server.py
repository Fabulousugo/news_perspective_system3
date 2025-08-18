# scripts/demo_server.py - Demo server with mock data for testing frontend

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import random
import time
from datetime import datetime, timedelta
import uvicorn
import click

# Demo data
DEMO_TOPICS = [
    "election", "climate", "economy", "healthcare", "immigration", 
    "education", "technology", "foreign policy", "energy", "trade"
]

DEMO_SOURCES = {
    "left-leaning": ["CNN", "The Guardian", "MSNBC", "NPR", "HuffPost"],
    "centrist": ["Reuters", "Associated Press", "BBC News", "Al Jazeera"],
    "right-leaning": ["Fox News", "New York Post", "Wall Street Journal", "Daily Wire", "Breitbart"]
}

DEMO_ARTICLES = {
    "election": {
        "left-leaning": [
            "Progressive coalition builds momentum for upcoming election",
            "Voting rights advocates push for expanded access",
            "Young voters energized by climate and social justice issues"
        ],
        "centrist": [
            "Polls show tight race in key battleground states", 
            "Election officials prepare for record turnout",
            "Bipartisan efforts focus on election security measures"
        ],
        "right-leaning": [
            "Conservative candidates gain ground in suburban districts",
            "Business community supports pro-growth election platform",
            "Traditional values voters mobilize for upcoming contests"
        ]
    },
    "climate": {
        "left-leaning": [
            "Scientists warn urgent action needed on climate crisis",
            "Green New Deal proposals gain traction in Congress",
            "Renewable energy investments show promising returns"
        ],
        "centrist": [
            "Climate policy proposals undergo bipartisan review",
            "Economic impact studies examine green transition costs",
            "International climate summit reaches compromise agreement"
        ],
        "right-leaning": [
            "Energy independence requires balanced approach to climate policy",
            "Job creation concerns arise from rapid green transition",
            "Market-based solutions offer practical climate alternatives"
        ]
    }
}

app = FastAPI(
    title="Demo News Perspective API",
    description="Demo server with mock data for testing the frontend (no API keys required)",
    version="1.0.0-demo"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PerspectiveRequest(BaseModel):
    query: str = ""
    days_back: int = 7
    optimization: str = "onnx"
    limit: int = 50

class BiasAnalysisRequest(BaseModel):
    text: str
    optimization: str = "onnx"

def generate_mock_articles(query: str, days_back: int) -> List[Dict]:
    """Generate mock perspective matches"""
    
    # Find relevant topic
    topic = "general"
    for demo_topic in DEMO_TOPICS:
        if demo_topic.lower() in query.lower():
            topic = demo_topic
            break
    
    matches = []
    num_matches = random.randint(3, 8)
    
    for i in range(num_matches):
        # Create a match with articles from different perspectives
        perspectives = {}
        
        # Select 2-3 random perspectives
        selected_biases = random.sample(list(DEMO_SOURCES.keys()), random.randint(2, 3))
        
        for bias in selected_biases:
            # Get demo articles for this topic and bias
            if topic in DEMO_ARTICLES and bias in DEMO_ARTICLES[topic]:
                article_titles = DEMO_ARTICLES[topic][bias]
                title = random.choice(article_titles)
            else:
                # Generic title
                title = f"{bias.replace('-', ' ').title()} perspective on {topic}"
            
            # Random source from this bias category
            source = random.choice(DEMO_SOURCES[bias])
            
            # Random publication time within the time range
            days_ago = random.randint(0, days_back)
            hours_ago = random.randint(0, 23)
            pub_time = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            
            # Bias label mapping
            bias_labels = {"left-leaning": 0, "centrist": 1, "right-leaning": 2}
            
            perspectives[bias] = {
                "title": title,
                "source": source,
                "url": f"https://example.com/{source.lower().replace(' ', '')}/article-{i}",
                "published_at": pub_time.isoformat(),
                "bias_label": bias_labels[bias]
            }
        
        # Create match
        match = {
            "story_id": f"demo_story_{i}",
            "topic": topic.title(),
            "confidence": random.uniform(0.75, 0.95),
            "perspectives": perspectives
        }
        matches.append(match)
    
    return matches

@app.get("/", response_class=HTMLResponse)
async def root():
    """Demo homepage"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head><title>Demo News Perspective API</title></head>
    <body style="font-family: Arial; margin: 40px; background: #f5f5f5;">
        <div style="max-width: 800px; background: white; padding: 30px; border-radius: 10px;">
            <h1>🎮 Demo News Perspective API</h1>
            <p><strong>Demo Mode:</strong> This server provides mock data for testing the frontend without requiring API keys.</p>
            
            <h2>Available Endpoints:</h2>
            <ul>
                <li><code>POST /perspectives/find</code> - Mock perspective matching</li>
                <li><code>POST /analyze/bias</code> - Mock bias analysis</li>
                <li><code>GET /health</code> - Health check</li>
                <li><code>GET /docs</code> - API documentation</li>
            </ul>
            
            <h2>Demo Features:</h2>
            <ul>
                <li>✅ Realistic mock data for testing</li>
                <li>✅ Simulated ONNX performance metrics</li>
                <li>✅ Multiple political perspectives</li>
                <li>✅ Temporal data for charts</li>
                <li>✅ No API keys required</li>
            </ul>
            
            <p><strong>Frontend:</strong> Open <code>frontend/index.html</code> and point it to this demo server.</p>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Demo health check"""
    return {
        "status": "healthy",
        "mode": "demo",
        "optimization_level": "onnx",
        "response_time_ms": random.uniform(5, 15),
        "models_loaded": {"demo": True},
        "performance": "simulated_optimized",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/perspectives/find")
async def find_perspectives(request: PerspectiveRequest):
    """Demo perspective finding with mock data"""
    
    # Simulate processing time (but much faster than real)
    start_time = time.time()
    await_time = random.uniform(0.5, 2.0)  # Simulate ONNX speed
    time.sleep(await_time)
    
    # Generate mock matches
    matches = generate_mock_articles(request.query, request.days_back)
    
    processing_time = time.time() - start_time
    
    # Mock collection summary
    collection_summary = {
        "left-leaning": random.randint(15, 35),
        "centrist": random.randint(8, 20), 
        "right-leaning": random.randint(20, 40)
    }
    
    total_articles = sum(collection_summary.values())
    
    # Mock performance stats
    performance = {
        "total_time": processing_time,
        "collection_time": processing_time * 0.6,
        "analysis_time": processing_time * 0.4,
        "articles_processed": total_articles,
        "matches_found": len(matches),
        "throughput": total_articles / processing_time,
        "optimization_level": "onnx"
    }
    
    return {
        "query": request.query,
        "matches": matches,
        "collection_summary": collection_summary,
        "performance": performance,
        "optimization": {
            "level": "onnx",
            "speedup_info": "3-6x faster than standard (simulated)"
        },
        "mode": "demo",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze/bias")
async def analyze_bias(request: BiasAnalysisRequest):
    """Demo bias analysis"""
    
    # Simulate processing
    start_time = time.time()
    time.sleep(random.uniform(0.1, 0.3))
    processing_time = time.time() - start_time
    
    # Mock bias prediction
    bias_options = ["left-leaning", "centrist", "right-leaning"]
    predicted_class = random.choice(bias_options)
    
    # Mock confidence based on text content
    confidence = random.uniform(0.7, 0.95)
    
    prediction = {
        "left-leaning": random.uniform(0.1, 0.4),
        "centrist": random.uniform(0.2, 0.5),
        "right-leaning": random.uniform(0.1, 0.4),
        "predicted_class": predicted_class,
        "confidence": confidence
    }
    
    return {
        "text_preview": request.text[:200] + "..." if len(request.text) > 200 else request.text,
        "bias_prediction": prediction,
        "performance": {
            "processing_time_ms": round(processing_time * 1000, 2),
            "optimization": "onnx",
            "speedup": "3-6x faster than standard (simulated)"
        },
        "mode": "demo",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/performance")
async def get_performance():
    """Demo performance stats"""
    return {
        "optimization_level": "onnx",
        "mode": "demo",
        "current_stats": {
            "bias_analysis_time": random.uniform(1.0, 3.0),
            "similarity_analysis_time": random.uniform(2.0, 5.0),
            "total_articles_processed": random.randint(50, 200),
            "perspective_matches_found": random.randint(5, 20),
            "using_source_bias": True
        },
        "system_info": {
            "models_loaded": {"demo": True},
            "optimization_benefits": {
                "onnx": "3-6x faster, 75% less memory (simulated)",
                "quantized": "2-4x faster, 50% less memory",
                "standard": "baseline performance"
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/benchmark")
async def run_benchmark():
    """Demo benchmark"""
    
    # Simulate benchmark
    time.sleep(1.0)
    
    return {
        "benchmark_results": {
            "optimization_level": "onnx",
            "test_articles": 15,
            "processing_time": random.uniform(0.5, 1.5),
            "throughput_articles_per_second": random.uniform(25, 45),
            "estimated_speedup": "3-6x vs standard (simulated)"
        },
        "mode": "demo",
        "note": "This is simulated data for testing the frontend",
        "timestamp": datetime.now().isoformat()
    }

@click.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
def main(host: str, port: int):
    """Start the demo server with mock data"""
    
    print("🎮 Starting Demo News Perspective Server")
    print("=" * 50)
    print("📌 Mode: Demo (mock data, no API keys required)")
    print(f"📡 Server: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print("🎯 Features: Mock perspective matching, simulated ONNX performance")
    print("")
    print("💡 Perfect for:")
    print("   • Testing the frontend without API keys")
    print("   • Demonstrating the system functionality") 
    print("   • Development and debugging")
    print("")
    print("🚀 To use with real data:")
    print("   python scripts/onnx_web_server.py --optimization onnx")
    print("")
    
    uvicorn.run(
        "demo_server:app",
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()