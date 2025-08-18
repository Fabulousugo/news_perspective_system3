# test_simple_extended.py - Quick test for the simple extended system

import sys,os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_simple_extended():
    """Test the simple extended collector"""
    print("🧪 Testing Simple Extended News Collection System")
    print("=" * 60)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from src.data_collection.simple_extended_collector import SimpleExtendedCollector
        from src.models.news_browser import NewsBrowser
        print("✅ Imports successful")
        
        # Test collector initialization
        print("\n🔧 Testing collector initialization...")
        collector = SimpleExtendedCollector()
        print("✅ Collector initialized")
        
        # Show source summary
        summary = collector.get_source_summary()
        print(f"\n📊 Source Summary:")
        print(f"   API sources: {summary['summary']['total_api_sources']}")
        print(f"   RSS sources: {summary['summary']['total_rss_sources']}")
        print(f"   Total sources: {summary['summary']['total_sources']}")
        
        print(f"\n📰 RSS Sources Available:")
        for bias, sources in summary['rss_sources'].items():
            if sources:
                print(f"   {bias}: {', '.join(sources)}")
        
        # Test basic collection
        print(f"\n📡 Testing article collection (small sample)...")
        articles = collector.collect_diverse_articles("election", days_back=3)
        
        total_articles = sum(len(arts) for arts in articles.values())
        print(f"✅ Collected {total_articles} articles")
        
        if total_articles > 0:
            print(f"\n📊 Collection breakdown:")
            for bias_category, bias_articles in articles.items():
                if bias_articles:
                    sources = list(set(a.source for a in bias_articles))
                    print(f"   {bias_category}: {len(bias_articles)} articles from {len(sources)} sources")
                    print(f"      Sources: {', '.join(sources)}")
            
            # Test browser
            print(f"\n🔍 Testing perspective browser...")
            browser = NewsBrowser()
            
            # Flatten articles
            all_articles = []
            for bias_arts in articles.values():
                all_articles.extend(bias_arts)
            
            # Test browsing (with limited articles to avoid timeout)
            browseable = browser.browse_articles(all_articles[:20])
            stats = browser.get_statistics(browseable)
            
            print(f"✅ Browser analysis complete:")
            print(f"   Articles analyzed: {stats['total_articles']}")
            print(f"   Articles with perspectives: {stats['articles_with_perspectives']}")
            print(f"   Perspective coverage: {stats['perspective_coverage']:.1%}")
            print(f"   Average perspectives per article: {stats['average_perspectives_per_article']:.1f}")
            print(f"   Max perspectives found: {stats['max_perspectives_found']}")
            
            if stats['articles_with_perspectives'] > 0:
                print(f"\n🎯 Example perspective matches:")
                count = 0
                for browseable_article in browseable:
                    if browseable_article.perspective_count > 0 and count < 3:
                        article = browseable_article.article
                        print(f"   📰 {article.title[:60]}...")
                        print(f"      Source: {article.source}")
                        print(f"      Perspectives: {browseable_article.perspective_count}")
                        
                        for related, similarity in browseable_article.related_articles[:2]:
                            bias_names = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
                            bias = bias_names.get(related.bias_label, "UNKNOWN")
                            print(f"         → {bias} ({similarity:.2f}): {related.title[:50]}... ({related.source})")
                        print()
                        count += 1
            
            print(f"🎉 System test completed successfully!")
            print(f"\n🚀 Ready to use! Try:")
            print(f"   python scripts/simple_enhanced_browser.py browse --query 'election' --days 7")
            
        else:
            print(f"⚠️  No articles collected. This might be due to:")
            print(f"   • API rate limits")
            print(f"   • Network connectivity")
            print(f"   • API keys not configured")
            print(f"\nTry checking your .env file and network connection")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"\nDebug info:")
        print(traceback.format_exc())
        
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check that you're in the project root directory")
        print(f"   2. Ensure your .env file has valid API keys")
        print(f"   3. Check internet connectivity")
        print(f"   4. Try the original system first: python scripts/run_application.py find-perspectives --query 'election'")
        
        return False

if __name__ == "__main__":
    test_simple_extended()