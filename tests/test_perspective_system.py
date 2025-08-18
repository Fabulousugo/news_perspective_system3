# test_perspective_system.py - Quick test for the fixed system

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

def test_system():
    """Test the perspective matching system with the fixes"""
    print("🧪 Testing News Perspective System...")
    
    try:
        # Import with error handling
        from src.data_collection.news_apis import NewsCollector
        from src.models.perspective_matcher import PerspectiveMatcher
        print("✅ Imports successful")
        
        # Test collector initialization
        print("🔧 Testing news collector...")
        collector = NewsCollector()
        print(f"✅ Collector initialized with APIs: {list(collector.apis.keys())}")
        
        # Test perspective matcher with source bias
        print("🔧 Testing perspective matcher...")
        matcher = PerspectiveMatcher(use_source_bias=True)
        print("✅ Perspective matcher initialized with source-based bias")
        
        # Test with a small sample
        print("🔬 Testing with sample articles...")
        diverse_articles = collector.collect_diverse_articles(
            query="election", 
            days_back=3
        )
        
        total_articles = sum(len(articles) for articles in diverse_articles.values())
        print(f"📊 Collected {total_articles} articles")
        
        if total_articles > 0:
            # Flatten articles
            all_articles = []
            for articles in diverse_articles.values():
                all_articles.extend(articles)
            
            # Test perspective matching
            print("🎯 Testing perspective matching...")
            matches = matcher.find_perspective_matches(all_articles[:20])  # Test with first 20
            
            print(f"✅ Found {len(matches)} perspective matches")
            
            if matches:
                print("📋 Sample match:")
                match = matches[0]
                print(f"   Topic: {match.topic}")
                print(f"   Confidence: {match.confidence:.3f}")
                for bias, article in match.articles.items():
                    print(f"   {bias}: {article.title[:60]}...")
        
        print("\n🎉 System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_system()
    if success:
        print("\n✅ Your system is ready to use!")
        print("Run: python scripts/run_application.py find-perspectives --query 'election'")
    else:
        print("\n❌ Please check the error messages above and fix any issues.")