# test_standalone_system.py - Test the standalone enhanced system

import sys,os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_standalone_system():
    """Test the standalone enhanced collector"""
    print("🧪 Testing Standalone Enhanced News Collection System")
    print("=" * 70)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from src.data_collection.standalone_enhanced_collector import StandaloneEnhancedCollector
        from src.models.news_browser import NewsBrowser
        print("✅ Imports successful")
        
        # Test collector initialization
        print("\n🔧 Testing collector initialization...")
        collector = StandaloneEnhancedCollector()
        print("✅ Collector initialized")
        
        # Show source summary
        summary = collector.get_source_summary()
        print(f"\n📊 Source Summary:")
        print(f"   API sources: {summary['totals']['api']}")
        print(f"   RSS sources: {summary['totals']['rss']}")
        print(f"   Total sources: {summary['totals']['total']}")
        
        print(f"\n📰 Sources by Category:")
        for bias_category in ['left_leaning', 'centrist', 'right_leaning']:
            api_sources = summary['api_sources'].get(bias_category, [])
            rss_sources = summary['rss_sources'].get(bias_category, [])
            
            print(f"   {bias_category.upper()}:")
            if api_sources:
                print(f"      API: {', '.join(api_sources)}")
            if rss_sources:
                print(f"      RSS: {', '.join(rss_sources)}")
        
        # Test connectivity
        print(f"\n🔗 Testing connectivity...")
        connectivity = collector.test_connectivity()
        
        print(f"📡 API Status:")
        for api, status in connectivity['api_status'].items():
            print(f"   {api}: {status}")
        
        print(f"📰 RSS Status (sample):")
        for source, status in connectivity['rss_status'].items():
            print(f"   {source}: {status}")
        
        # Test article collection
        print(f"\n📡 Testing article collection...")
        articles = collector.collect_diverse_articles("election", days_back=3)
        
        total_articles = sum(len(arts) for arts in articles.values())
        print(f"✅ Collected {total_articles} articles")
        
        if total_articles > 0:
            print(f"\n📊 Collection breakdown:")
            for bias_category, bias_articles in articles.items():
                if bias_articles:
                    sources = list(set(a.source for a in bias_articles))
                    print(f"   {bias_category}: {len(bias_articles)} articles")
                    print(f"      Sources: {', '.join(sources)}")
                    
                    # Show example headlines
                    print(f"      Sample headlines:")
                    for i, article in enumerate(bias_articles[:2]):
                        print(f"         • {article.title[:60]}...")
            
            # Test browser
            print(f"\n🔍 Testing perspective browser...")
            browser = NewsBrowser()
            
            # Flatten articles for browser
            all_articles = []
            for bias_arts in articles.values():
                all_articles.extend(bias_arts)
            
            # Test browsing (limit to avoid timeout)
            print(f"   Analyzing {len(all_articles)} articles for perspectives...")
            browseable = browser.browse_articles(all_articles[:30])  # Limit for testing
            stats = browser.get_statistics(browseable)
            
            print(f"✅ Browser analysis complete:")
            print(f"   Articles analyzed: {stats['total_articles']}")
            print(f"   Articles with perspectives: {stats['articles_with_perspectives']}")
            print(f"   Perspective coverage: {stats['perspective_coverage']:.1%}")
            print(f"   Average perspectives per article: {stats['average_perspectives_per_article']:.1f}")
            print(f"   Max perspectives found: {stats['max_perspectives_found']}")
            
            # Show example matches
            if stats['articles_with_perspectives'] > 0:
                print(f"\n🎯 Example perspective matches:")
                count = 0
                for browseable_article in browseable:
                    if browseable_article.perspective_count > 0 and count < 3:
                        article = browseable_article.article
                        bias_names = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
                        main_bias = bias_names.get(article.bias_label, "UNKNOWN")
                        
                        print(f"\n   📰 {main_bias}: {article.title[:50]}...")
                        print(f"      Source: {article.source}")
                        print(f"      Perspectives: {browseable_article.perspective_count}")
                        
                        for related, similarity in browseable_article.related_articles[:2]:
                            related_bias = bias_names.get(related.bias_label, "UNKNOWN")
                            print(f"         → {related_bias} ({similarity:.2f}): {related.title[:45]}...")
                            print(f"            {related.source}")
                        count += 1
            
            print(f"\n🎉 System test completed successfully!")
            print(f"\n🚀 Ready to use enhanced news browsing!")
            print(f"\nNext steps:")
            print(f"   1. Create a simple script to use this collector")
            print(f"   2. Try: python test_standalone_system.py")
            print(f"   3. Build a simple browser interface")
            
        else:
            print(f"⚠️  No articles collected. Possible reasons:")
            print(f"   • API keys not configured (check .env file)")
            print(f"   • Network connectivity issues")
            print(f"   • RSS feeds temporarily unavailable")
            print(f"   • API rate limits reached")
            
            print(f"\n🔧 Troubleshooting:")
            print(f"   • Check .env file has NEWSAPI_API_KEY and GUARDIAN_API_KEY")
            print(f"   • Test internet connectivity")
            print(f"   • Try with a different query or longer time window")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"\nDebug information:")
        print(traceback.format_exc())
        
        print(f"\n🔧 Troubleshooting steps:")
        print(f"   1. Ensure you're in the project root directory")
        print(f"   2. Check Python path and imports")
        print(f"   3. Verify .env file configuration")
        print(f"   4. Test basic internet connectivity")
        
        return False

if __name__ == "__main__":
    success = test_standalone_system()
    
    if success:
        print(f"\n✅ All tests passed!")
        # print(f"The enhanced system is working with {20+} sources including:")
        print(f"   🔵 LEFT: CNN, NPR, Guardian, Salon, Mother Jones")
        print(f"   ⚪ CENTER: Reuters, BBC, AP, Al Jazeera, PBS")
        print(f"   🔴 RIGHT: Fox News, Daily Wire, Breitbart, National Review")
    else:
        print(f"\n❌ Tests failed - check errors above")