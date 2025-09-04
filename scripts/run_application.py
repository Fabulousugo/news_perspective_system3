# Main application script
# scripts/run_application.py

import asyncio
import uvicorn
import click
import logging
from pathlib import Path
import sys,os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.news_apis import NewsCollector
from src.models.perspective_matcher import PerspectiveMatcher
from config.settings import settings

# Main application script
# scripts/run_application.py

import asyncio
import uvicorn
import click
import logging
import os
from pathlib import Path
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_collection.news_apis import NewsCollector
from src.models.perspective_matcher import PerspectiveMatcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """News Perspective Diversification System"""
    pass

@cli.command()
@click.option('--query', '-q', default='', help='Search query for articles')
@click.option('--days', '-d', default=7, help='Number of days to look back')
@click.option('--min-perspectives', '-p', default=2, help='Minimum perspectives required')
@click.option('--debug', is_flag=True, help='Enable debug output')
def find_perspectives(query: str, days: int, min_perspectives: int, debug: bool):
    """Find articles covering the same story from different perspectives"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Finding perspectives for query: '{query}'")
    
    # Initialize components
    try:
        collector = NewsCollector()
        if not collector.apis:
            print("❌ No API clients initialized!")
            print("Run: python scripts/run_application.py debug")
            return
            
        print(f"✅ Initialized with APIs: {list(collector.apis.keys())}")
        
    except Exception as e:
        print(f"❌ Failed to initialize collector: {e}")
        return
    
    # Collect articles
    print(f"🔍 Collecting articles for: {query or 'general news'}")
    print(f"   Looking back {days} days...")
    
    try:
        diverse_articles = collector.collect_diverse_articles(
            query=query,
            days_back=days
        )
    except Exception as e:
        print(f"❌ Failed to collect articles: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return
    
    # Print collection summary
    total_articles = sum(len(articles) for articles in diverse_articles.values())
    print(f"\n📊 Collected {total_articles} articles:")
    for bias, articles in diverse_articles.items():
        print(f"   📰 {bias}: {len(articles)} articles")
        if debug and articles:
            for i, article in enumerate(articles[:3]):  # Show first 3 in debug mode
                print(f"      {i+1}. {article.title[:60]}... ({article.source})")
    
    if total_articles == 0:
        print("\n❌ No articles found!")
        print("Possible reasons:")
        print("   • API keys not configured properly")
        print("   • Query too specific")
        print("   • Rate limits exceeded")
        print("   • No recent news matching criteria")
        print("\nTry:")
        print("   • python scripts/run_application.py debug")
        print("   • python scripts/setup_wizard.py") 
        print("   • Use broader query or empty query ('')")
        return
    
    # Try perspective matching only if we have articles
    try:
        matcher = PerspectiveMatcher()
        
        # Flatten articles
        all_articles = []
        for articles in diverse_articles.values():
            all_articles.extend(articles)
        
        # Find perspective matches
        print(f"\n🔍 Finding perspective matches...")
        matches = matcher.find_perspective_matches(
            all_articles,
            min_perspectives=min_perspectives
        )
        
        # Display results
        if matches:
            print(f"\n✅ Found {len(matches)} perspective matches:\n")
            
            for i, match in enumerate(matches):
                print(f"📖 Match {i+1}: {match.topic}")
                print(f"   Confidence: {match.confidence:.3f}")
                print("   Perspectives:")
                
                for bias, article in match.articles.items():
                    print(f"      • {bias.upper()}: {article.title}")
                    print(f"        Source: {article.source}")
                    if debug:
                        print(f"        URL: {article.url}")
                print("-" * 80)
            
            # Summary
            summary = matcher.get_perspective_summary(matches)
            print(f"\n📊 Summary:")
            print(f"   Total matches: {summary['total_matches']}")
            print(f"   Average confidence: {summary['average_confidence']:.3f}")
            print(f"   Perspective distribution: {summary['perspective_distribution']}")
        else:
            print(f"\n⚠️  No perspective matches found")
            print("This could mean:")
            print("   • Articles are too diverse (not covering same stories)")
            print("   • Similarity threshold too high")
            print("   • Not enough articles per bias category")
            print("   • Try with different query or more days")
            
    except Exception as e:
        print(f"❌ Failed during perspective matching: {e}")
        if debug:
            import traceback
            traceback.print_exc()

@cli.command()
@click.option('--text', '-t', required=True, help='Text to analyze for bias')
def analyze_bias(text: str):
    """Analyze political bias of a text"""
    classifier = BiasClassifier(load_pretrained=True)
    
    prediction = classifier.predict_single(text)
    
    print(f"\nBias Analysis Results:")
    print(f"Text: {text[:100]}...")
    print(f"\nPredicted bias: {prediction['predicted_class']}")
    print(f"Confidence: {prediction['confidence']:.3f}")
    print(f"\nAll probabilities:")
    for bias, prob in prediction.items():
        if bias not in ['predicted_class', 'confidence']:
            print(f"  {bias}: {prob:.3f}")

@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind the server to')
@click.option('--port', default=8000, help='Port to bind the server to')
def serve(host: str, port: int):
    """Start the API server"""
    print(f"Starting News Perspective API server...")
    print(f"Server will be available at: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "src.api.routes:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

@cli.command()
def debug():
    """Debug the system configuration and API connectivity"""
    print("🔍 DEBUGGING NEWS PERSPECTIVE SYSTEM")
    print("=" * 50)
    
    # Check environment variables
    print("\n1. Environment Variables:")
    newsapi_key = os.getenv('NEWSAPI_API_KEY')
    guardian_key = os.getenv('GUARDIAN_API_KEY')
    
    print(f"   NewsAPI key: {'✅ Found' if newsapi_key else '❌ Missing'}")
    print(f"   Guardian key: {'✅ Found' if guardian_key else '❌ Missing'}")
    
    if not (newsapi_key or guardian_key):
        print("\n❌ No API keys found!")
        print("Create a .env file with:")
        print("NEWSAPI_API_KEY=your_key_here")
        print("GUARDIAN_API_KEY=your_key_here")
        return
    
    # Test NewsCollector
    print("\n2. Testing NewsCollector:")
    try:
        collector = NewsCollector()
        print(f"   Available APIs: {list(collector.apis.keys())}")
        
        if collector.apis:
            print("   ✅ NewsCollector initialized successfully")
            
            # Test collection
            print("\n3. Testing Article Collection:")
            results = collector.collect_diverse_articles(query="", days_back=1)
            
            total_articles = sum(len(articles) for articles in results.values())
            print(f"   Total articles collected: {total_articles}")
            
            for bias_type, articles in results.items():
                print(f"   {bias_type}: {len(articles)} articles")
                if articles:
                    print(f"      Sample: {articles[0].title[:60]}...")
            
            if total_articles > 0:
                print("\n✅ System is working correctly!")
            else:
                print("\n⚠️  No articles collected. This could be due to:")
                print("   - API rate limits")
                print("   - No recent articles matching criteria")
                print("   - API service issues")
        else:
            print("   ❌ No APIs initialized")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

@cli.command()
def setup():
    """Set up the application environment"""
    print("Setting up News Perspective Diversification System...")
    
    # Create necessary directories
    directories = [
        Path("data"),
        Path("data/models"),
        Path("data/raw"), 
        Path("data/processed")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Run the setup wizard:")
    print("   python scripts/setup_wizard.py")
    print("2. Or manually set up environment variables:")
    print("   Create .env file with your API keys")
    print("3. Test the system:")
    print("   python scripts/run_application.py debug")
    print("4. Find perspectives:")
    print("   python scripts/run_application.py find-perspectives --query 'election'")

@cli.command()
def quick_test():
    """Quick test to verify the system is working"""
    print("🚀 QUICK SYSTEM TEST")
    print("=" * 40)
    
    # Test 1: Environment
    print("1. Checking environment...")
    newsapi_key = os.getenv('NEWSAPI_API_KEY')
    if newsapi_key:
        print("   ✅ NewsAPI key found")
    else:
        print("   ❌ NewsAPI key missing")
        print("   Run: python scripts/setup_wizard.py")
        return
    
    # Test 2: Basic API call
    print("2. Testing API connection...")
    import requests
    try:
        response = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={
                'apiKey': newsapi_key,
                'country': 'us',
                'pageSize': 3
            },
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                print(f"   ✅ API working! Found {len(articles)} articles")
                if articles:
                    print(f"   📰 Sample: {articles[0]['title'][:50]}...")
            else:
                print(f"   ❌ API error: {data.get('message')}")
                return
        else:
            print(f"   ❌ HTTP error: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return
    
    # Test 3: NewsCollector
    print("3. Testing NewsCollector...")
    try:
        collector = NewsCollector()
        if collector.apis:
            print(f"   ✅ Collector ready with: {list(collector.apis.keys())}")
        else:
            print("   ❌ Collector not initialized")
            return
    except Exception as e:
        print(f"   ❌ Collector failed: {e}")
        return
    
    print("\n🎉 Quick test passed! System is ready.")
    print("Try: python scripts/run_application.py find-perspectives --query 'news'")

if __name__ == "__main__":
    cli()

