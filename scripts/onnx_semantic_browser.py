# scripts/onnx_semantic_browser.py - ONNX-optimized browser with enhanced semantic search

import click
import logging
import time
from pathlib import Path
import sys,os
from datetime import datetime
import webbrowser

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.simple_extended_collector import SimpleExtendedCollector
from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
from src.search.semantic_search import EnhancedSemanticSearch, SmartQueryProcessor, SearchResult
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """🚀 ONNX-Optimized News Browser with Enhanced Semantic Search"""
    pass

@cli.command()
@click.option('--query', '-q', required=True, help='Search query (e.g., "AI", "Biden", "climate change")')
@click.option('--days', '-d', default=7, help='Days to look back')
@click.option('--optimization', '-o', type=click.Choice(['standard', 'quantized', 'onnx']), 
              default='onnx', help='Optimization level')
@click.option('--limit', '-l', default=20, help='Number of articles to show')
@click.option('--semantic', is_flag=True, default=True, help='Use enhanced semantic search')
def search(query: str, days: int, optimization: str, limit: int, semantic: bool):
    """Enhanced semantic search with ONNX optimization"""
    
    print(f"🔍 Enhanced Semantic News Search (ONNX Optimized)")
    print("=" * 60)
    print(f"🎯 Query: '{query}'")
    print(f"📅 Days back: {days}")
    print(f"⚡ Optimization: {optimization}")
    print(f"🧠 Semantic search: {'Enabled' if semantic else 'Disabled'}")
    print("")
    
    try:
        # Initialize components
        start_init = time.time()
        print("🚀 Initializing enhanced search system...")
        
        collector = SimpleExtendedCollector()
        search_engine = EnhancedSemanticSearch() if semantic else None
        query_processor = SmartQueryProcessor()
        matcher = OptimizedPerspectiveMatcher(optimization_level=optimization)
        
        init_time = time.time() - start_init
        print(f"✅ System initialized in {init_time:.2f}s")
        
        # Process query for better results
        processed_query = query_processor.process_query(query)
        if processed_query != query.lower():
            print(f"🔧 Query processed: '{query}' → '{processed_query}'")
        
        # Collect articles
        print(f"\n📡 Collecting articles...")
        start_collection = time.time()
        
        diverse_articles = collector.collect_diverse_articles(
            query=processed_query,
            days_back=days
        )
        
        collection_time = time.time() - start_collection
        
        # Flatten articles
        all_articles = []
        for bias_articles in diverse_articles.values():
            all_articles.extend(bias_articles)
        
        print(f"✅ Collected {len(all_articles)} articles in {collection_time:.2f}s")
        
        if len(all_articles) == 0:
            print("❌ No articles found. Try:")
            print("   • Broader search terms (e.g., 'technology' instead of specific tech)")
            print("   • Different time range (--days 14)")
            print("   • Check your API keys")
            return
        
        # Enhanced semantic search
        print(f"\n🧠 Performing enhanced semantic search...")
        start_search = time.time()
        
        if semantic and search_engine:
            search_results = search_engine.search_articles(all_articles, query, top_k=limit*2)
            relevant_articles = [result.article for result in search_results]
            
            # Show search insights
            if search_results:
                avg_relevance = sum(r.relevance_score for r in search_results) / len(search_results)
                avg_semantic = sum(r.semantic_similarity for r in search_results) / len(search_results)
                
                print(f"✅ Semantic search completed:")
                print(f"   Found {len(search_results)} highly relevant articles")
                print(f"   Average relevance: {avg_relevance:.3f}")
                print(f"   Average semantic similarity: {avg_semantic:.3f}")
                
                # Show query expansion info
                expanded_terms = search_engine.query_expander.expand_query(query)
                if len(expanded_terms) > 1:
                    print(f"   Query expanded to: {', '.join(list(expanded_terms.keys())[:5])}")
        else:
            # Fallback to simple keyword search
            relevant_articles = [a for a in all_articles if query.lower() in a.title.lower() or 
                               (a.description and query.lower() in a.description.lower())]
            print(f"✅ Keyword search found {len(relevant_articles)} articles")
        
        search_time = time.time() - start_search
        
        if not relevant_articles:
            print(f"\n❌ No relevant articles found for '{query}'")
            print(f"💡 Search suggestions:")
            print(f"   • Try synonyms (e.g., 'AI' vs 'artificial intelligence')")
            print(f"   • Use broader terms")
            print(f"   • Check spelling")
            return
        
        # Display results with enhanced information
        print(f"\n📋 Search Results (showing top {min(limit, len(relevant_articles))}):")
        print("=" * 100)
        
        bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
        
        display_results = relevant_articles[:limit] if not semantic else search_results[:limit]
        
        for i, item in enumerate(display_results):
            if semantic and isinstance(item, SearchResult):
                article = item.article
                relevance = item.relevance_score
                semantic_sim = item.semantic_similarity
                matched_terms = item.matched_terms[:3]  # Top 3 matched terms
            else:
                article = item if not semantic else item.article
                relevance = 1.0
                semantic_sim = 1.0
                matched_terms = [query]
            
            # Format article info
            time_str = article.published_at.strftime("%m/%d %H:%M")
            bias_indicator = bias_names.get(article.bias_label, "❓ UNKNOWN")
            
            print(f"\n[{i+1:2d}] {bias_indicator} | {time_str}")
            if semantic:
                print(f"     📊 Relevance: {relevance:.3f} | Semantic: {semantic_sim:.3f}")
                print(f"     🏷️  Matched: {', '.join(matched_terms)}")
            
            print(f"     📰 {article.title}")
            print(f"     🏢 {article.source}")
            
            if article.description:
                desc_preview = article.description[:100]
                if len(article.description) > 100:
                    desc_preview += "..."
                print(f"     📄 {desc_preview}")
            
            print(f"     🔗 {article.url}")
            print("     " + "─" * 90)
        
        # Find perspectives on search results
        if len(relevant_articles) >= 2:
            print(f"\n🎯 Finding perspectives on '{query}' coverage...")
            start_perspective = time.time()
            
            perspective_articles = relevant_articles[:30]  # Use top 30 for perspective analysis
            matches = matcher.find_perspective_matches_fast(perspective_articles)
            
            perspective_time = time.time() - start_perspective
            
            if matches:
                print(f"✅ Found {len(matches)} perspective matches in {perspective_time:.2f}s:")
                print("-" * 80)
                
                for i, match in enumerate(matches[:5]):  # Show top 5 perspective matches
                    print(f"\n🎯 Perspective Match {i+1}: {match.topic}")
                    print(f"   Confidence: {match.confidence:.3f}")
                    
                    for bias_category, match_article in match.articles.items():
                        bias_icon = bias_names.get(match_article.bias_label, "❓")
                        print(f"   {bias_icon} {bias_category.upper()}: {match_article.title}")
                        print(f"      📰 {match_article.source}")
        
        # Performance summary
        total_time = init_time + collection_time + search_time
        print(f"\n📊 Performance Summary (ONNX Optimized):")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Collection: {collection_time:.2f}s")
        print(f"   Semantic search: {search_time:.2f}s") 
        print(f"   Articles processed: {len(all_articles)}")
        print(f"   Relevant results: {len(relevant_articles)}")
        print(f"   Optimization: {optimization}")
        
        # Interactive options
        print(f"\n💡 Interactive Options:")
        print(f"   • Enter article number (1-{len(display_results)}) to open in browser")
        print(f"   • Type 'related' to find related search terms")
        print(f"   • Type 'perspectives' to see all perspective matches")
        print(f"   • Type 'exit' to quit")
        
        while True:
            try:
                user_input = input("\n> ").strip().lower()
                
                if user_input == 'exit':
                    break
                elif user_input == 'related':
                    _show_related_terms(search_engine, query)
                elif user_input == 'perspectives':
                    if 'matches' in locals():
                        _show_all_perspectives(matches, bias_names)
                    else:
                        print("No perspective matches found")
                elif user_input.isdigit():
                    article_num = int(user_input) - 1
                    if 0 <= article_num < len(display_results):
                        if semantic:
                            article_to_open = display_results[article_num].article if isinstance(display_results[article_num], SearchResult) else display_results[article_num]
                        else:
                            article_to_open = display_results[article_num]
                        print(f"🌐 Opening: {article_to_open.title}")
                        webbrowser.open(article_to_open.url)
                    else:
                        print("❌ Invalid article number")
                else:
                    print("❌ Unknown command. Try: article number, 'related', 'perspectives', or 'exit'")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        logger.error(f"Enhanced search failed: {e}")
        import traceback
        print(f"Debug: {traceback.format_exc()}")

def _show_related_terms(search_engine, query):
    """Show related search terms and synonyms"""
    print(f"\n🔗 Related Terms for '{query}':")
    print("-" * 40)
    
    expanded_terms = search_engine.query_expander.expand_query(query)
    exclusions = search_engine.query_expander.get_exclusions(query)
    
    print(f"📈 Synonyms and expansions:")
    for term, weight in sorted(expanded_terms.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {term} (weight: {weight:.1f})")
    
    if exclusions:
        print(f"\n🚫 Excluded terms (to avoid confusion):")
        for term in exclusions:
            print(f"   • {term}")
    
    print(f"\n💡 Try these related searches:")
    related_searches = list(expanded_terms.keys())[:5]
    for term in related_searches:
        if term != query.lower():
            print(f"   python scripts/onnx_semantic_browser.py search --query \"{term}\"")

def _show_all_perspectives(matches, bias_names):
    """Show detailed perspective analysis"""
    print(f"\n🎯 All Perspective Matches:")
    print("=" * 60)
    
    for i, match in enumerate(matches):
        print(f"\n📰 Story {i+1}: {match.topic}")
        print(f"🎯 Confidence: {match.confidence:.3f}")
        print(f"📅 Detected: {match.timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        print(f"📊 Different Perspectives:")
        for bias_category, article in match.articles.items():
            bias_icon = bias_names.get(article.bias_label, "❓")
            print(f"   {bias_icon} {bias_category.upper()}")
            print(f"      📰 {article.title}")
            print(f"      🏢 {article.source}")
            print(f"      🕒 {article.published_at.strftime('%m/%d %H:%M')}")
        
        if i < len(matches) - 1:
            print("   " + "─" * 50)

@cli.command()
@click.option('--query1', required=True, help='First query to compare')
@click.option('--query2', required=True, help='Second query to compare')
@click.option('--days', '-d', default=7, help='Days to look back')
def compare(query1: str, query2: str, days: int):
    """Compare semantic search results between two queries"""
    
    print(f"🔄 Comparing Search Queries")
    print("=" * 40)
    print(f"Query 1: '{query1}'")
    print(f"Query 2: '{query2}'")
    print("")
    
    try:
        # Initialize
        collector = SimpleExtendedCollector()
        search_engine = EnhancedSemanticSearch()
        
        # Collect articles
        print("📡 Collecting articles...")
        diverse_articles = collector.collect_diverse_articles("", days_back=days)
        
        all_articles = []
        for bias_articles in diverse_articles.values():
            all_articles.extend(bias_articles)
        
        print(f"✅ Analyzing {len(all_articles)} articles")
        
        # Search for both queries
        print(f"\n🔍 Searching for '{query1}'...")
        results1 = search_engine.search_articles(all_articles, query1, top_k=10)
        
        print(f"🔍 Searching for '{query2}'...")
        results2 = search_engine.search_articles(all_articles, query2, top_k=10)
        
        # Compare results
        print(f"\n📊 Comparison Results:")
        print(f"   '{query1}': {len(results1)} relevant articles")
        print(f"   '{query2}': {len(results2)} relevant articles")
        
        # Find overlap
        urls1 = {r.article.url for r in results1}
        urls2 = {r.article.url for r in results2}
        overlap = urls1.intersection(urls2)
        
        print(f"   Overlap: {len(overlap)} articles")
        
        if overlap:
            print(f"\n🔗 Articles covering both topics:")
            for result in results1:
                if result.article.url in overlap:
                    print(f"   • {result.article.title} ({result.article.source})")
        
        # Show unique results
        print(f"\n📰 Unique to '{query1}':")
        for result in results1[:3]:
            if result.article.url not in overlap:
                print(f"   • {result.article.title} (relevance: {result.relevance_score:.3f})")
        
        print(f"\n📰 Unique to '{query2}':")
        for result in results2[:3]:
            if result.article.url not in overlap:
                print(f"   • {result.article.title} (relevance: {result.relevance_score:.3f})")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

@cli.command()
def test_semantic():
    """Test semantic search improvements"""
    
    print("🧪 Testing Semantic Search Improvements")
    print("=" * 50)
    
    # Test cases that were problematic
    test_cases = [
        ("AI", "artificial intelligence"),
        ("Biden", "Trump"),  # Should not overlap
        ("climate change", "global warming"),
        ("crypto", "cryptocurrency"),
        ("healthcare", "health care")
    ]
    
    search_engine = EnhancedSemanticSearch()
    
    for query1, query2 in test_cases:
        print(f"\n🔍 Testing: '{query1}' vs '{query2}'")
        
        # Test query expansion
        expanded1 = search_engine.query_expander.expand_query(query1)
        expanded2 = search_engine.query_expander.expand_query(query2)
        
        # Find common terms
        common_terms = set(expanded1.keys()).intersection(set(expanded2.keys()))
        
        print(f"   '{query1}' expands to: {list(expanded1.keys())[:3]}")
        print(f"   '{query2}' expands to: {list(expanded2.keys())[:3]}")
        print(f"   Common terms: {list(common_terms)}")
        
        # Test exclusions
        exclusions1 = search_engine.query_expander.get_exclusions(query1)
        exclusions2 = search_engine.query_expander.get_exclusions(query2)
        
        if exclusions1:
            print(f"   '{query1}' excludes: {exclusions1}")
        if exclusions2:
            print(f"   '{query2}' excludes: {exclusions2}")
    
    print(f"\n✅ Semantic search test completed!")

if __name__ == "__main__":
    cli()