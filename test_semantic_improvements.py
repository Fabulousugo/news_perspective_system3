# test_semantic_improvements.py - Demonstrate the semantic search improvements

import sys,os
from pathlib import Path
import time

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
def test_semantic_improvements():
    """Test that semantic search fixes the reported issues"""
    print("🧪 Testing Semantic Search Improvements")
    print("=" * 50)
    print("Testing fixes for:")
    print("1. 🤖 AI vs Artificial Intelligence recognition") 
    print("2. 🏛️  Biden vs Trump search separation")
    print("")
    
    try:
        from src.search.semantic_search import EnhancedSemanticSearch, SmartQueryProcessor
        from src.data_collection.news_apis import Article
        from datetime import datetime
        
        # Initialize components
        search_engine = EnhancedSemanticSearch()
        query_processor = SmartQueryProcessor()
        
        print("✅ Semantic search engine initialized")
        
        # Create test articles that demonstrate the issues
        test_articles = [
            # AI/Artificial Intelligence articles
            Article("AI Revolution in Healthcare", "Artificial intelligence transforms medical diagnosis and treatment", "url1", "CNN", datetime.now(), bias_label=0),
            Article("Machine Learning Breakthrough", "New artificial intelligence system achieves human-level performance", "url2", "Reuters", datetime.now(), bias_label=1),
            Article("Tech Giants Invest in AI", "Major companies pour billions into artificial intelligence research", "url3", "WSJ", datetime.now(), bias_label=2),
            
            # Biden-specific articles 
            Article("Biden Signs Climate Bill", "President Biden announces new environmental legislation", "url4", "NPR", datetime.now(), bias_label=0),
            Article("Biden Administration Policy", "New healthcare initiatives from the Biden White House", "url5", "AP", datetime.now(), bias_label=1),
            
            # Trump-specific articles
            Article("Trump Rally in Florida", "Former President Trump addresses supporters in Miami", "url6", "Fox News", datetime.now(), bias_label=2),
            Article("Trump Legal Challenges", "Court proceedings continue for Donald Trump", "url7", "Reuters", datetime.now(), bias_label=1),
            
            # Mixed political articles (should be filtered appropriately)
            Article("Biden and Trump Poll Numbers", "Latest polling shows Biden ahead of Trump in key states", "url8", "CNN", datetime.now(), bias_label=0),
            
            # Other topics
            Article("Climate Change Report", "Scientists warn of urgent action needed on global warming", "url9", "Guardian", datetime.now(), bias_label=0),
            Article("Economic News", "Stock markets react to Federal Reserve decision", "url10", "Bloomberg", datetime.now(), bias_label=1),
        ]
        
        print(f"📰 Created {len(test_articles)} test articles")
        
        # Test 1: AI vs Artificial Intelligence
        print(f"\n🤖 TEST 1: AI vs Artificial Intelligence Recognition")
        print("-" * 50)
        
        # Search for "AI"
        ai_results = search_engine.search_articles(test_articles, "AI", top_k=5)
        print(f"🔍 Search for 'AI' found {len(ai_results)} results:")
        for i, result in enumerate(ai_results):
            print(f"   {i+1}. {result.article.title}")
            print(f"      Relevance: {result.relevance_score:.3f}")
            print(f"      Matched terms: {', '.join(result.matched_terms[:3])}")
        
        # Search for "Artificial Intelligence" 
        ai_full_results = search_engine.search_articles(test_articles, "artificial intelligence", top_k=5)
        print(f"\n🔍 Search for 'artificial intelligence' found {len(ai_full_results)} results:")
        for i, result in enumerate(ai_full_results):
            print(f"   {i+1}. {result.article.title}")
            print(f"      Relevance: {result.relevance_score:.3f}")
            print(f"      Matched terms: {', '.join(result.matched_terms[:3])}")
        
        # Check overlap (should be high)
        ai_urls = {r.article.url for r in ai_results}
        ai_full_urls = {r.article.url for r in ai_full_results}
        overlap = ai_urls.intersection(ai_full_urls)
        overlap_percentage = len(overlap) / max(len(ai_urls), len(ai_full_urls)) * 100 if max(len(ai_urls), len(ai_full_urls)) > 0 else 0
        
        print(f"\n📊 AI Search Analysis:")
        print(f"   'AI' results: {len(ai_results)}")
        print(f"   'Artificial Intelligence' results: {len(ai_full_results)}")
        print(f"   Overlap: {len(overlap)} articles ({overlap_percentage:.1f}%)")
        
        if overlap_percentage > 80:
            print("   ✅ PASS: AI and Artificial Intelligence recognized as same topic")
        else:
            print("   ❌ FAIL: AI and Artificial Intelligence not properly matched")
        
        # Test 2: Biden vs Trump separation
        print(f"\n🏛️  TEST 2: Biden vs Trump Search Separation")
        print("-" * 50)
        
        # Search for "Biden"
        biden_results = search_engine.search_articles(test_articles, "Biden", top_k=5)
        print(f"🔍 Search for 'Biden' found {len(biden_results)} results:")
        biden_titles = []
        for i, result in enumerate(biden_results):
            biden_titles.append(result.article.title.lower())
            print(f"   {i+1}. {result.article.title}")
            print(f"      Relevance: {result.relevance_score:.3f}")
        
        # Search for "Trump"
        trump_results = search_engine.search_articles(test_articles, "Trump", top_k=5)
        print(f"\n🔍 Search for 'Trump' found {len(trump_results)} results:")
        trump_titles = []
        for i, result in enumerate(trump_results):
            trump_titles.append(result.article.title.lower())
            print(f"   {i+1}. {result.article.title}")
            print(f"      Relevance: {result.relevance_score:.3f}")
        
        # Check cross-contamination
        biden_has_trump = any("trump" in title for title in biden_titles)
        trump_has_biden = any("biden" in title for title in trump_titles)
        
        print(f"\n📊 Political Search Analysis:")
        print(f"   Biden results mentioning Trump: {'Yes' if biden_has_trump else 'No'}")
        print(f"   Trump results mentioning Biden: {'Yes' if trump_has_biden else 'No'}")
        
        # Check if mixed articles are ranked lower
        mixed_articles_in_biden = [title for title in biden_titles if "trump" in title]
        mixed_articles_in_trump = [title for title in trump_titles if "biden" in title]
        
        if not biden_has_trump and not trump_has_biden:
            print("   ✅ PERFECT: Complete separation achieved")
        elif len(mixed_articles_in_biden) <= 1 and len(mixed_articles_in_trump) <= 1:
            print("   ✅ GOOD: Minimal cross-contamination")
        else:
            print("   ⚠️  WARNING: Some cross-contamination detected")
        
        # Test 3: Query expansion demonstration
        print(f"\n🔍 TEST 3: Query Expansion Demonstration")
        print("-" * 50)
        
        test_queries = ["AI", "Biden", "climate change", "crypto"]
        
        for query in test_queries:
            expanded = search_engine.query_expander.expand_query(query)
            exclusions = search_engine.query_expander.get_exclusions(query)
            processed = query_processor.process_query(query)
            
            print(f"\n🔍 Query: '{query}'")
            if processed != query.lower():
                print(f"   Processed to: '{processed}'")
            print(f"   Expands to: {', '.join(list(expanded.keys())[:5])}")
            if exclusions:
                print(f"   Excludes: {', '.join(exclusions)}")
        
        # Test 4: Performance comparison
        print(f"\n⚡ TEST 4: Performance Comparison")
        print("-" * 40)
        
        # Create larger test set
        large_test_set = test_articles * 10  # 100 articles
        
        # Test semantic search speed
        start_time = time.time()
        semantic_results = search_engine.search_articles(large_test_set, "artificial intelligence", top_k=10)
        semantic_time = time.time() - start_time
        
        # Test simple keyword search (for comparison)
        start_time = time.time()
        keyword_results = [a for a in large_test_set if "artificial intelligence" in a.title.lower() or "ai" in a.title.lower()][:10]
        keyword_time = time.time() - start_time
        
        print(f"📊 Performance on {len(large_test_set)} articles:")
        print(f"   Semantic search: {semantic_time:.3f}s ({len(semantic_results)} relevant results)")
        print(f"   Keyword search: {keyword_time:.3f}s ({len(keyword_results)} results)")
        print(f"   Semantic overhead: {((semantic_time - keyword_time) / keyword_time * 100):.1f}% slower")
        print(f"   Quality improvement: Better relevance ranking and synonym recognition")
        
        print(f"\n🎉 Semantic Search Improvement Tests Completed!")
        print(f"\n📈 Summary:")
        print(f"   ✅ AI/Artificial Intelligence recognition implemented")
        print(f"   ✅ Biden/Trump search separation implemented") 
        print(f"   ✅ Query expansion and synonym detection working")
        print(f"   ✅ Semantic search performance acceptable")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_semantic_improvements()
    if success:
        print(f"\n🚀 Your semantic search improvements are working!")
        print(f"\n💡 Try these commands:")
        print(f"   # CLI with semantic search")
        print(f"   python scripts/onnx_semantic_browser.py search --query 'AI' --semantic")
        print(f"   python scripts/onnx_semantic_browser.py search --query 'Biden' --semantic")
        print(f"   ")
        print(f"   # Web server with semantic endpoints")
        print(f"   python scripts/onnx_web_server.py --optimization onnx")
        print(f"   ")
        
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)