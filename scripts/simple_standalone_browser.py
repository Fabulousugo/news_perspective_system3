# scripts/simple_standalone_browser.py - Simple browser using standalone collector

import sys,os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
def browse_news(query="", days=7, limit=15):
    """Simple news browsing function"""
    
    print(f"📰 Enhanced News Browser (Standalone)")
    print("=" * 50)
    print(f"🔍 Query: {query or 'General news'}")
    print(f"📅 Days back: {days}")
    print("")
    
    try:
        from src.data_collection.standalone_enhanced_collector import StandaloneEnhancedCollector
        from src.models.news_browser import NewsBrowser
        
        # Initialize
        print("🚀 Initializing enhanced collector...")
        collector = StandaloneEnhancedCollector()
        browser = NewsBrowser()
        
        # Show source info
        summary = collector.get_source_summary()
        print(f"📊 Using {summary['totals']['total']} sources:")
        print(f"   API: {summary['totals']['api']} sources")
        print(f"   RSS: {summary['totals']['rss']} sources")
        print("")
        
        # Collect articles
        print("📡 Collecting articles...")
        articles = collector.collect_diverse_articles(query, days)
        
        # Flatten for browser
        all_articles = []
        total_collected = 0
        for bias_category, bias_articles in articles.items():
            all_articles.extend(bias_articles)
            total_collected += len(bias_articles)
            if bias_articles:
                sources = list(set(a.source for a in bias_articles))
                print(f"   {bias_category}: {len(bias_articles)} articles from {len(sources)} sources")
        
        print(f"✅ Total collected: {total_collected} articles")
        
        if total_collected == 0:
            print("❌ No articles found. Try:")
            print("   • Broader search query")
            print("   • More days (e.g., 14)")
            print("   • Check API keys in .env")
            return
        
        # Analyze for perspectives
        print(f"\n🔍 Analyzing for perspectives...")
        browseable = browser.browse_articles(all_articles)
        stats = browser.get_statistics(browseable)
        
        print(f"📊 Analysis Results:")
        print(f"   Articles with perspectives: {stats['articles_with_perspectives']}")
        print(f"   Perspective coverage: {stats['perspective_coverage']:.1%}")
        print(f"   Average perspectives: {stats['average_perspectives_per_article']:.1f}")
        print(f"   Max perspectives found: {stats['max_perspectives_found']}")
        
        # Show articles with perspectives
        print(f"\n📋 Top Articles with Perspectives (showing {limit}):")
        print("=" * 80)
        
        # Sort by perspective count
        browseable.sort(key=lambda x: x.perspective_count, reverse=True)
        
        bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
        
        for i, article_data in enumerate(browseable[:limit]):
            article = article_data.article
            bias_indicator = bias_names.get(article.bias_label, "❓")
            
            print(f"\n[{i+1:2d}] {bias_indicator} | 🎯 {article_data.perspective_count} perspectives")
            print(f"     📰 {article.title}")
            print(f"     🏢 {article.source}")
            print(f"     🕒 {article.published_at.strftime('%m/%d %H:%M')}")
            
            if article_data.perspective_count > 0:
                print(f"     🔗 Alternative perspectives:")
                for j, (related, similarity) in enumerate(article_data.related_articles[:3]):
                    related_bias = bias_names.get(related.bias_label, "❓")
                    print(f"        {related_bias} ({similarity:.2f}) {related.title[:50]}...")
                    print(f"           📰 {related.source}")
            
            print("     " + "─" * 70)
        
        # Interactive options
        print(f"\n🎯 Interactive Options:")
        print(f"   Enter article number (1-{min(limit, len(browseable))}) for details")
        print(f"   Type 'search KEYWORD' to filter")
        print(f"   Type 'stats' for detailed statistics")
        print(f"   Type 'sources' for source breakdown")
        print(f"   Type 'exit' to quit")
        
        while True:
            try:
                user_input = input("\n> ").strip().lower()
                
                if user_input == 'exit':
                    break
                elif user_input == 'stats':
                    show_detailed_stats(stats, browseable)
                elif user_input == 'sources':
                    show_source_breakdown(articles, summary)
                elif user_input.startswith('search '):
                    keyword = user_input[7:]
                    search_articles(browseable, keyword)
                elif user_input.isdigit():
                    article_num = int(user_input) - 1
                    if 0 <= article_num < len(browseable):
                        show_article_details(browseable[article_num])
                    else:
                        print("❌ Invalid article number")
                else:
                    print("❌ Unknown command")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())

def show_detailed_stats(stats, browseable):
    """Show detailed statistics"""
    print("\n📊 Detailed Statistics:")
    print("=" * 40)
    
    perspective_counts = [a.perspective_count for a in browseable]
    
    print(f"Total articles: {stats['total_articles']}")
    print(f"Articles with perspectives: {stats['articles_with_perspectives']}")
    print(f"Perspective coverage: {stats['perspective_coverage']:.1%}")
    
    print(f"\nPerspective Distribution:")
    print(f"   0 perspectives: {len([p for p in perspective_counts if p == 0])}")
    print(f"   1-2 perspectives: {len([p for p in perspective_counts if 1 <= p <= 2])}")
    print(f"   3-5 perspectives: {len([p for p in perspective_counts if 3 <= p <= 5])}")
    print(f"   6+ perspectives: {len([p for p in perspective_counts if p >= 6])}")
    
    print(f"\nBias Distribution:")
    for bias, count in stats['bias_distribution'].items():
        print(f"   {bias}: {count} articles")

def show_source_breakdown(articles, summary):
    """Show source breakdown"""
    print("\n📰 Source Breakdown:")
    print("=" * 40)
    
    print(f"Available Sources:")
    for bias_category in ['left_leaning', 'centrist', 'right_leaning']:
        api_sources = summary['api_sources'].get(bias_category, [])
        rss_sources = summary['rss_sources'].get(bias_category, [])
        
        print(f"\n{bias_category.upper()}:")
        if api_sources:
            print(f"   API: {', '.join(api_sources)}")
        if rss_sources:
            print(f"   RSS: {', '.join(rss_sources)}")
    
    print(f"\nArticles Collected:")
    for bias_category, bias_articles in articles.items():
        if bias_articles:
            source_counts = {}
            for article in bias_articles:
                source_counts[article.source] = source_counts.get(article.source, 0) + 1
            
            print(f"\n{bias_category.upper()} ({len(bias_articles)} articles):")
            for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {source}: {count} articles")

def search_articles(browseable, keyword):
    """Search through articles"""
    matching = []
    for article_data in browseable:
        if (keyword in article_data.article.title.lower() or 
            (article_data.article.description and keyword in article_data.article.description.lower())):
            matching.append(article_data)
    
    print(f"\n🔍 Found {len(matching)} articles matching '{keyword}':")
    
    bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
    
    for i, article_data in enumerate(matching[:10]):
        article = article_data.article
        bias_indicator = bias_names.get(article.bias_label, "❓")
        print(f"   [{i+1}] {bias_indicator} | 🎯 {article_data.perspective_count} perspectives")
        print(f"       📰 {article.title}")
        print(f"       🏢 {article.source}")

def show_article_details(article_data):
    """Show full article details"""
    article = article_data.article
    
    print("\n" + "=" * 60)
    print("ARTICLE DETAILS")
    print("=" * 60)
    
    bias_names = {0: "🔵 LEFT-LEANING", 1: "⚪ CENTRIST", 2: "🔴 RIGHT-LEANING"}
    bias_indicator = bias_names.get(article.bias_label, "❓ UNKNOWN")
    
    print(f"Title: {article.title}")
    print(f"Source: {article.source} ({bias_indicator})")
    print(f"Published: {article.published_at.strftime('%Y-%m-%d %H:%M')}")
    print(f"Author: {article.author or 'Not specified'}")
    print(f"URL: {article.url}")
    
    if article.description:
        print(f"\nDescription: {article.description}")
    
    if article.content:
        content = article.content[:300]
        if len(article.content) > 300:
            content += "... [truncated]"
        print(f"\nContent: {content}")
    
    print(f"\n🎯 Alternative Perspectives ({article_data.perspective_count}):")
    if article_data.perspective_count > 0:
        for i, (related, similarity) in enumerate(article_data.related_articles):
            related_bias = bias_names.get(related.bias_label, "❓ UNKNOWN")
            print(f"\n   [{i+1}] {related_bias} - Similarity: {similarity:.1%}")
            print(f"       📰 {related.title}")
            print(f"       🏢 {related.source}")
            print(f"       🔗 {related.url}")
    else:
        print("   No alternative perspectives found")
    
    print("=" * 60)

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced News Perspective Browser")
    parser.add_argument('--query', '-q', default='', help='Search query')
    parser.add_argument('--days', '-d', type=int, default=7, help='Days to look back')
    parser.add_argument('--limit', '-l', type=int, default=15, help='Number of articles to show')
    
    args = parser.parse_args()
    
    browse_news(args.query, args.days, args.limit)