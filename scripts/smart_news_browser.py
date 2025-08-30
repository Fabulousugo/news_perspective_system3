# scripts/smart_news_browser.py - CLI with enhanced semantic search

import click
import logging
from pathlib import Path
import sys,os
from datetime import datetime
import time

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.simple_extended_collector import SimpleExtendedCollector
from src.models.enhanced_news_browser import SmartNewsBrowser
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """🧠 Smart News Browser with Semantic Search"""
    pass

@cli.command()
@click.option('--query', '-q', required=True, help='Search query (e.g., "AI", "Biden", "climate")')
@click.option('--days', '-d', default=7, help='Days to look back')
@click.option('--limit', '-l', default=20, help='Number of results to show')
@click.option('--optimization', '-o', type=click.Choice(['standard', 'quantized', 'onnx']), 
              default='quantized', help='Optimization level')
@click.option('--perspectives/--no-perspectives', default=True, help='Find alternative perspectives')
def search(query: str, days: int, limit: int, optimization: str, perspectives: bool):
    """Smart search with semantic understanding and query expansion"""
    
    print(f"🧠 Smart News Search")
    print("=" * 50)
    print(f"🔍 Query: '{query}'")
    print(f"📅 Looking back: {days} days")
    print(f"⚡ Optimization: {optimization}")
    print(f"🎭 Find perspectives: {'Yes' if perspectives else 'No'}")
    print("")
    
    try:
        # Initialize components
        print("🚀 Initializing smart search system...")
        collector = SimpleExtendedCollector()
        browser = SmartNewsBrowser(optimization_level=optimization)
        
        # Collect articles
        print("📡 Collecting articles from diverse sources...")
        start_collection = time.time()
        
        diverse_articles = collector.collect_diverse_articles(
            query="",  # Collect all articles, then search semantically
            days_back=days
        )
        
        # Flatten articles
        all_articles = []
        for bias_articles in diverse_articles.values():
            all_articles.extend(bias_articles)
        
        collection_time = time.time() - start_collection
        print(f"✅ Collected {len(all_articles)} articles in {collection_time:.2f}s")
        
        if not all_articles:
            print("❌ No articles found. Check your API keys.")
            return
        
        # Smart search
        print(f"\n🧠 Performing semantic search for '{query}'...")
        start_search = time.time()
        
        results = browser.smart_search(all_articles, query, find_perspectives=perspectives)
        insights = browser.get_search_insights(results, query)
        
        search_time = time.time() - start_search
        total_time = collection_time + search_time
        
        # Display insights
        print(f"\n📊 Search Insights:")
        print(f"   🔄 Query expanded: {query} → {', '.join(insights['query_analysis']['expanded_terms'])}")
        print(f"   🏷️  Query type: {insights['query_analysis']['query_type']}")
        print(f"   📈 Results found: {insights['result_stats']['total_results']}")
        print(f"   🎭 With perspectives: {insights['result_stats']['results_with_perspectives']}")
        print(f"   ⚡ Search time: {search_time:.2f}s")
        print(f"   📊 Avg relevance: {insights['result_stats']['avg_relevance']:.2f}")
        
        # Source distribution
        if insights['source_distribution']:
            print(f"\n📰 Source Distribution:")
            for source, count in sorted(insights['source_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
                print(f"   📄 {source}: {count} articles")
        
        # Bias distribution  
        if insights['bias_distribution']:
            print(f"\n🏛️  Political Balance:")
            for bias, count in insights['bias_distribution'].items():
                print(f"   {bias}: {count} articles")
        
        # Show results
        if results:
            print(f"\n🎯 Search Results (top {min(limit, len(results))}):")
            print("=" * 100)
            
            bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
            
            for i, result in enumerate(results[:limit]):
                article = result.article
                
                # Format display
                time_str = article.published_at.strftime("%m/%d %H:%M")
                bias_indicator = bias_names.get(article.bias_label, "❓ UNKNOWN")
                
                # Perspective indicator
                perspective_info = ""
                if result.perspective_count > 0:
                    perspective_info = f" | 🎭 {result.perspective_count} perspectives"
                
                print(f"\n[{i+1:2d}] {bias_indicator} | {time_str} | Relevance: {result.relevance_score:.2f}{perspective_info}")
                print(f"     📰 {article.title}")
                print(f"     🏢 {article.source}")
                print(f"     🔗 Matches: {', '.join(result.query_matches)}")
                
                # Show alternative perspectives
                if result.related_articles:
                    print(f"     🎭 Alternative perspectives:")
                    for j, (related_article, similarity) in enumerate(result.related_articles[:2]):
                        related_bias = bias_names.get(related_article.bias_label, "❓")
                        print(f"        {related_bias} ({similarity:.2f}): {related_article.title[:60]}...")
                        print(f"           📰 {related_article.source}")
                
                print("     " + "─" * 90)
            
            # Interactive options
            print(f"\n🎮 Interactive Options:")
            print(f"   • Enter result number (1-{min(limit, len(results))}) to see full article")
            print(f"   • Type 'related X' to find articles related to result X")
            print(f"   • Type 'suggestions' to see search suggestions")
            print(f"   • Type 'refine NEWQUERY' to search with different terms")
            print(f"   • Type 'exit' to quit")
            
            while True:
                try:
                    user_input = input("\n> ").strip().lower()
                    
                    if user_input == 'exit':
                        break
                    elif user_input == 'suggestions':
                        suggestions = insights.get('search_suggestions', [])
                        if suggestions:
                            print(f"\n💡 Search suggestions:")
                            for i, suggestion in enumerate(suggestions, 1):
                                print(f"   {i}. {suggestion}")
                        else:
                            print("No suggestions available")
                    elif user_input.startswith('refine '):
                        new_query = user_input[7:]
                        print(f"\n🔄 Refining search with '{new_query}'...")
                        # Recursive call with new query
                        search.main(['-q', new_query, '-d', str(days), '-l', str(limit)], 
                                  standalone_mode=False)
                        break
                    elif user_input.startswith('related '):
                        try:
                            result_num = int(user_input.split()[1]) - 1
                            if 0 <= result_num < len(results):
                                _show_related_articles(browser, results[result_num].article, all_articles)
                            else:
                                print("❌ Invalid result number")
                        except (ValueError, IndexError):
                            print("❌ Invalid format. Use: related X")
                    elif user_input.isdigit():
                        result_num = int(user_input) - 1
                        if 0 <= result_num < len(results):
                            _show_full_result(results[result_num])
                        else:
                            print("❌ Invalid result number")
                    else:
                        print("❌ Unknown command. Try: result number, 'related X', 'suggestions', 'refine QUERY', or 'exit'")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        else:
            print(f"\n❌ No results found for '{query}'")
            
            # Show suggestions
            suggestions = insights.get('search_suggestions', [])
            if suggestions:
                print(f"\n💡 Try these related searches:")
                for suggestion in suggestions[:5]:
                    print(f"   • {suggestion}")
            
            print(f"\n🔧 Tips:")
            print(f"   • Try broader terms (e.g., 'climate' instead of 'climate change legislation')")
            print(f"   • Use key topics (e.g., 'AI', 'Biden', 'economy')")
            print(f"   • Check spelling and try synonyms")
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        logger.error(f"Smart search failed: {e}")
        import traceback
        print(f"Debug: {traceback.format_exc()}")

def _show_full_result(result):
    """Show full details of a search result"""
    article = result.article
    
    print("\n" + "=" * 80)
    print("📰 FULL ARTICLE DETAILS")
    print("=" * 80)
    print(f"Title: {article.title}")
    print(f"Source: {article.source}")
    print(f"Published: {article.published_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"Relevance Score: {result.relevance_score:.2f}")
    print(f"Query Matches: {', '.join(result.query_matches)}")
    print(f"URL: {article.url}")
    
    if article.description:
        print(f"\nDescription:")
        print(f"{article.description}")
    
    if article.content:
        print(f"\nContent Preview:")
        content_preview = article.content[:500]
        if len(article.content) > 500:
            content_preview += "... [content truncated]"
        print(f"{content_preview}")
    
    if result.related_articles:
        print(f"\n🎭 Alternative Perspectives ({len(result.related_articles)}):")
        bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
        
        for i, (related, similarity) in enumerate(result.related_articles):
            bias = bias_names.get(related.bias_label, "❓")
            print(f"   {i+1}. {bias} ({similarity:.2f}) - {related.source}")
            print(f"      {related.title}")
            print(f"      {related.url}")
    
    print("=" * 80)

def _show_related_articles(browser, article, all_articles):
    """Show articles related to a specific article"""
    print(f"\n🔗 Finding articles related to:")
    print(f"   {article.title}")
    print(f"   Source: {article.source}")
    
    related = browser.find_related_stories(article, all_articles, max_results=5)
    
    if related:
        print(f"\n📊 Related articles ({len(related)}):")
        bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
        
        for i, (related_article, similarity) in enumerate(related):
            bias = bias_names.get(related_article.bias_label, "❓")
            print(f"   {i+1}. {bias} ({similarity:.2f}) - {related_article.source}")
            print(f"      {related_article.title}")
            print(f"      Published: {related_article.published_at.strftime('%m/%d %H:%M')}")
    else:
        print("   No closely related articles found")

@cli.command()
@click.option('--articles', '-a', default=100, help='Number of articles to test with')
def test_search(articles: int):
    """Test the enhanced search system with problematic queries"""
    
    print("🧪 Enhanced Search System Test")
    print("=" * 40)
    
    try:
        # Collect test articles
        print("📡 Collecting test articles...")
        collector = SimpleExtendedCollector()
        browser = SmartNewsBrowser()
        
        diverse_articles = collector.collect_diverse_articles("", days_back=7)
        all_articles = []
        for bias_articles in diverse_articles.values():
            all_articles.extend(bias_articles)
        
        test_articles = all_articles[:articles]
        print(f"✅ Using {len(test_articles)} articles for testing")
        
        # Test problematic queries
        problematic_queries = [
            ("ai", "Should find AI and Artificial Intelligence articles"),
            ("artificial intelligence", "Should also find AI articles"), 
            ("biden", "Should focus on Biden, not include Trump articles"),
            ("trump", "Should focus on Trump, not include Biden articles"),
            ("crypto", "Should find cryptocurrency and bitcoin articles"),
            ("climate change", "Should find climate and environmental articles")
        ]
        
        print(f"\n🔍 Testing Enhanced Search:")
        print("-" * 60)
        
        for query, expected in problematic_queries:
            print(f"\nQuery: '{query}'")
            print(f"Expected: {expected}")
            
            # Perform search
            start_time = time.time()
            results = browser.smart_search(test_articles, query, find_perspectives=False)
            search_time = time.time() - start_time
            
            insights = browser.get_search_insights(results, query)
            
            print(f"Results: {len(results)} articles ({search_time:.3f}s)")
            print(f"Expanded: {', '.join(insights['query_analysis']['expanded_terms'])}")
            
            # Show top results
            for i, result in enumerate(results[:3]):
                print(f"  {i+1}. [{result.relevance_score:.2f}] {result.article.title[:60]}...")
                print(f"     Source: {result.article.source} | Matches: {', '.join(result.query_matches)}")
            
            print("-" * 40)
        
        print("\n✅ Enhanced search test completed!")
        print("\n🎯 Key improvements:")
        print("   🔹 Query expansion: AI ↔ Artificial Intelligence") 
        print("   🔹 Entity detection: Biden queries focus on Biden")
        print("   🔹 Cross-contamination filtering: Trump ≠ Biden")
        print("   🔹 Relevance scoring: Main subject gets priority")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())

@cli.command()
@click.argument('query1')
@click.argument('query2')
def compare(query1: str, query2: str):
    """Compare search results for two different queries"""
    
    print(f"🔄 Comparing Search Results")
    print("=" * 40)
    print(f"Query 1: '{query1}'")
    print(f"Query 2: '{query2}'")
    print("")
    
    try:
        # Collect articles
        collector = SimpleExtendedCollector()
        browser = SmartNewsBrowser()
        
        diverse_articles = collector.collect_diverse_articles("", days_back=7)
        all_articles = []
        for bias_articles in diverse_articles.values():
            all_articles.extend(bias_articles)
        
        # Search both queries
        results1 = browser.smart_search(all_articles, query1, find_perspectives=False)
        results2 = browser.smart_search(all_articles, query2, find_perspectives=False)
        
        insights1 = browser.get_search_insights(results1, query1)
        insights2 = browser.get_search_insights(results2, query2)
        
        # Compare results
        print(f"📊 Comparison Results:")
        print(f"   {query1}: {len(results1)} results")
        print(f"   {query2}: {len(results2)} results")
        print("")
        
        print(f"🔄 Query Expansions:")
        print(f"   {query1} → {', '.join(insights1['query_analysis']['expanded_terms'])}")
        print(f"   {query2} → {', '.join(insights2['query_analysis']['expanded_terms'])}")
        print("")
        
        # Check for overlapping results
        urls1 = {r.article.url for r in results1}
        urls2 = {r.article.url for r in results2}
        overlap = len(urls1.intersection(urls2))
        
        print(f"📈 Result Analysis:")
        print(f"   Overlapping articles: {overlap}")
        print(f"   Unique to '{query1}': {len(urls1 - urls2)}")
        print(f"   Unique to '{query2}': {len(urls2 - urls1)}")
        
        # Show top results for each
        for query, results in [(query1, results1), (query2, results2)]:
            print(f"\n🔍 Top results for '{query}':")
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. [{result.relevance_score:.2f}] {result.article.title[:50]}...")
                print(f"      {result.article.source}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    cli()