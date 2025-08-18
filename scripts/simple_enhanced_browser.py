# scripts/simple_enhanced_browser.py - Bulletproof enhanced browser

import click
import logging
from pathlib import Path
import sys,os
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.simple_extended_collector import SimpleExtendedCollector
from src.models.news_browser import NewsBrowser, ArticleWithPerspectives
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """📰 Enhanced News Perspective Browser (Simple & Reliable)"""
    pass

@cli.command()
@click.option('--query', '-q', default='', help='Search query (leave empty for general news)')
@click.option('--days', '-d', default=7, help='Days to look back (more days = more perspectives)')
@click.option('--sort', '-s', type=click.Choice(['recent', 'diverse']), default='diverse', 
              help='Sort by: recent (newest first) or diverse (most perspectives first)')
@click.option('--limit', '-l', default=20, help='Number of articles to show')
def browse(query: str, days: int, sort: str, limit: int):
    """Browse news articles with enhanced source diversity"""
    
    print(f"📰 Enhanced News Perspective Browser")
    print("=" * 70)
    print(f"🔍 Query: {query or 'General news'}")
    print(f"📅 Looking back: {days} days")
    print(f"🔄 Sort by: {sort}")
    print("")
    
    try:
        # Initialize components
        print("🚀 Initializing enhanced news collection...")
        collector = SimpleExtendedCollector()
        browser = NewsBrowser()
        
        # Show source summary
        source_summary = collector.get_source_summary()
        print(f"📊 Available sources:")
        print(f"   API sources: {source_summary['summary']['total_api_sources']} (NewsAPI, Guardian)")
        print(f"   RSS sources: {source_summary['summary']['total_rss_sources']} (Daily Wire, Breitbart, Salon, etc.)")
        print(f"   Total sources: {source_summary['summary']['total_sources']}")
        print("")
        
        # Collect articles
        print("📡 Collecting articles from diverse sources...")
        diverse_articles = collector.collect_diverse_articles(
            query=query,
            days_back=days
        )
        
        # Flatten and combine
        all_articles = []
        for bias_category, articles in diverse_articles.items():
            all_articles.extend(articles)
        
        print(f"✅ Collected {len(all_articles)} articles")
        
        if len(all_articles) == 0:
            print("❌ No articles found. Try:")
            print("   • A broader search query")
            print("   • More days (--days 14)")
            print("   • Check your API keys")
            return
        
        # Show collection breakdown
        print(f"\n📊 Collection breakdown:")
        for bias_category, articles in diverse_articles.items():
            if articles:
                sources = list(set(a.source for a in articles))
                print(f"   {bias_category}: {len(articles)} articles from {len(sources)} sources")
                source_names = ', '.join(sources)
                if len(source_names) > 80:
                    source_names = source_names[:77] + "..."
                print(f"      Sources: {source_names}")
        
        # Browse articles
        print(f"\n🔍 Analyzing for perspectives...")
        browseable_articles = browser.browse_articles(all_articles, sort_by=sort)
        
        # Show enhanced statistics
        stats = browser.get_statistics(browseable_articles)
        print(f"\n📊 Enhanced Statistics:")
        print(f"   📰 Total articles: {stats['total_articles']}")
        print(f"   🎯 Articles with perspectives: {stats['articles_with_perspectives']}")
        print(f"   📈 Perspective coverage: {stats['perspective_coverage']:.1%}")
        print(f"   🔗 Average perspectives per article: {stats['average_perspectives_per_article']:.1f}")
        print(f"   🏆 Max perspectives found: {stats['max_perspectives_found']}")
        print(f"   ⚙️  Similarity thresholds: loose={stats['similarity_thresholds']['loose']}, tight={stats['similarity_thresholds']['tight']}")
        
        print(f"\n🏛️  Bias distribution:")
        for bias, count in stats['bias_distribution'].items():
            print(f"   {bias}: {count} articles")
        
        # Show articles
        print(f"\n📋 Articles (showing top {limit}):")
        print("=" * 120)
        
        for i, browseable in enumerate(browseable_articles[:limit]):
            article = browseable.article
            
            # Format publish time
            time_str = article.published_at.strftime("%m/%d %H:%M")
            
            # Bias indicator
            bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT", 3: "🟡 LIBERTARIAN"}
            bias_indicator = bias_names.get(article.bias_label, "❓ UNKNOWN")
            
            # Enhanced perspective indicator
            if browseable.perspective_count > 0:
                perspective_indicator = f"🎯 {browseable.perspective_count} perspectives"
                # Show which bias categories have perspectives
                perspective_biases = set()
                for related_article, _ in browseable.related_articles:
                    perspective_biases.add(bias_names.get(related_article.bias_label, "❓"))
                perspective_detail = f"({', '.join(perspective_biases)})"
            else:
                perspective_indicator = "🔍 No perspectives found"
                perspective_detail = ""
            
            print(f"\n[{i+1:2d}] {bias_indicator} | {time_str} | {perspective_indicator} {perspective_detail}")
            print(f"     📰 {article.title}")
            print(f"     🏢 {article.source}")
            
            # Show top perspectives if available
            if browseable.perspective_count > 0:
                print(f"     🔗 Alternative perspectives:")
                shown_perspectives = 0
                for j, (related_article, similarity) in enumerate(browseable.related_articles):
                    if shown_perspectives >= 3:  # Limit to top 3
                        break
                    related_bias = bias_names.get(related_article.bias_label, "❓")
                    if related_bias != bias_indicator:  # Only show different perspectives
                        print(f"        {related_bias} ({similarity:.2f}) {related_article.title[:55]}...")
                        print(f"           📰 {related_article.source}")
                        shown_perspectives += 1
                
                if browseable.perspective_count > shown_perspectives:
                    remaining = browseable.perspective_count - shown_perspectives
                    print(f"        ... and {remaining} more perspectives")
            
            print("     " + "─" * 110)
        
        # Interactive menu
        if browseable_articles:
            print(f"\n🎯 Interactive Options:")
            print(f"   • Enter article number (1-{min(limit, len(browseable_articles))}) to read full article")
            print(f"   • Type 'perspectives X' to see all perspectives for article X")
            print(f"   • Type 'search KEYWORD' to filter articles")
            print(f"   • Type 'sources' to see source breakdown")
            print(f"   • Type 'exit' to quit")
            
            while True:
                try:
                    user_input = input("\n> ").strip().lower()
                    
                    if user_input == 'exit':
                        break
                    elif user_input == 'sources':
                        _show_sources(diverse_articles, source_summary)
                    elif user_input.startswith('search '):
                        keyword = user_input[7:]
                        _search_articles(browser, all_articles, keyword)
                    elif user_input.startswith('perspectives '):
                        try:
                            article_num = int(user_input.split()[1]) - 1
                            if 0 <= article_num < len(browseable_articles):
                                _show_article_perspectives(browseable_articles[article_num])
                            else:
                                print("❌ Invalid article number")
                        except (ValueError, IndexError):
                            print("❌ Invalid format. Use: perspectives X")
                    elif user_input.isdigit():
                        article_num = int(user_input) - 1
                        if 0 <= article_num < len(browseable_articles):
                            _show_full_article(browseable_articles[article_num])
                        else:
                            print("❌ Invalid article number")
                    else:
                        print("❌ Unknown command. Try: article number, 'perspectives X', 'search KEYWORD', 'sources', or 'exit'")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Browse command failed: {e}")
        import traceback
        print(f"Debug info: {traceback.format_exc()}")

def _show_sources(diverse_articles, source_summary):
    """Show source breakdown"""
    print("\n" + "📊" * 40)
    print("SOURCE BREAKDOWN")
    print("📊" * 40)
    
    print(f"\n📋 Source Types:")
    print(f"   API sources: {', '.join(source_summary['api_sources'])}")
    for bias, sources in source_summary['rss_sources'].items():
        if sources:
            print(f"   RSS {bias}: {', '.join(sources)}")
    
    print(f"\n📰 Articles by Source:")
    for bias_category, articles in diverse_articles.items():
        if not articles:
            continue
            
        print(f"\n🏛️  {bias_category.upper()} ({len(articles)} articles):")
        
        # Count articles per source
        source_counts = {}
        for article in articles:
            source_counts[article.source] = source_counts.get(article.source, 0) + 1
        
        # Show source breakdown
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   📰 {source}: {count} articles")
    
    print("📊" * 40)

def _show_full_article(browseable: ArticleWithPerspectives):
    """Show full article details"""
    article = browseable.article
    
    print("\n" + "=" * 100)
    print(f"📰 FULL ARTICLE")
    print("=" * 100)
    print(f"Title: {article.title}")
    print(f"Source: {article.source}")
    print(f"Author: {article.author or 'Not specified'}")
    print(f"Published: {article.published_at.strftime('%Y-%m-%d %H:%M')}")
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
    
    print(f"\n🎯 Perspectives Available: {browseable.perspective_count}")
    
    if browseable.perspective_count > 0:
        print(f"Alternative viewpoints:")
        bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
        for i, (related_article, similarity) in enumerate(browseable.related_articles):
            bias_indicator = bias_names.get(related_article.bias_label, "❓")
            print(f"  {i+1}. {bias_indicator} ({similarity:.2f}) {related_article.title}")
            print(f"     📰 {related_article.source}")
    
    print("=" * 100)

def _show_article_perspectives(browseable: ArticleWithPerspectives):
    """Show all perspectives for an article"""
    article = browseable.article
    
    print("\n" + "🎯" * 50)
    print(f"PERSPECTIVES FOR: {article.title}")
    print("🎯" * 50)
    
    if browseable.perspective_count == 0:
        print("❌ No alternative perspectives found for this article")
        return
    
    bias_names = {0: "🔵 LEFT-LEANING", 1: "⚪ CENTRIST", 2: "🔴 RIGHT-LEANING"}
    
    print(f"\n📰 Original Article:")
    print(f"   {bias_names.get(article.bias_label, '❓ UNKNOWN')} - {article.source}")
    print(f"   {article.title}")
    
    print(f"\n🔄 Alternative Perspectives ({browseable.perspective_count}):")
    
    for i, (related_article, similarity) in enumerate(browseable.related_articles):
        bias_indicator = bias_names.get(related_article.bias_label, "❓ UNKNOWN")
        print(f"\n   [{i+1}] {bias_indicator} - Similarity: {similarity:.1%}")
        print(f"       📰 {related_article.source}")
        print(f"       📝 {related_article.title}")
        print(f"       🕒 {related_article.published_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"       🔗 {related_article.url}")
        
        if related_article.description:
            desc_preview = related_article.description[:150]
            if len(related_article.description) > 150:
                desc_preview += "..."
            print(f"       📄 {desc_preview}")
    
    print("\n" + "🎯" * 50)

def _search_articles(browser: NewsBrowser, all_articles, keyword: str):
    """Search and filter articles"""
    print(f"\n🔍 Searching for '{keyword}'...")
    
    matching_articles = browser.search_articles(all_articles, keyword)
    
    if not matching_articles:
        print(f"❌ No articles found matching '{keyword}'")
        return
    
    print(f"✅ Found {len(matching_articles)} articles matching '{keyword}':")
    print("-" * 80)
    
    bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
    
    for i, browseable in enumerate(matching_articles[:10]):  # Show top 10
        article = browseable.article
        bias_indicator = bias_names.get(article.bias_label, "❓")
        
        print(f"[{i+1}] {bias_indicator} | 🎯 {browseable.perspective_count} perspectives")
        print(f"    📰 {article.title}")
        print(f"    🏢 {article.source}")
        print()

@cli.command()
def sources():
    """Show available sources and test connectivity"""
    print("📰 Enhanced Source Overview")
    print("=" * 50)
    
    try:
        collector = SimpleExtendedCollector()
        summary = collector.get_source_summary()
        
        print(f"📊 Source Summary:")
        print(f"   Total sources: {summary['summary']['total_sources']}")
        print(f"   API sources: {summary['summary']['total_api_sources']}")
        print(f"   RSS sources: {summary['summary']['total_rss_sources']}")
        
        print(f"\n📡 API Sources:")
        for api in summary['api_sources']:
            print(f"   ✅ {api}")
        
        print(f"\n📰 RSS Sources by Political Bias:")
        for bias_category, sources in summary['rss_sources'].items():
            if sources:
                print(f"   🏛️  {bias_category.upper()}:")
                for source in sources:
                    print(f"      📄 {source}")
        
        print(f"\n💡 This gives you access to sources like:")
        print(f"   🔵 LEFT: CNN, Guardian, NPR, Salon, Mother Jones, HuffPost")
        print(f"   ⚪ CENTER: Reuters, BBC, AP, Al Jazeera, PBS")
        print(f"   🔴 RIGHT: Fox News, Daily Wire, Breitbart, National Review, Federalist")
        
    except Exception as e:
        print(f"❌ Error: {e}")

@cli.command()
@click.option('--threshold', '-t', default=0.65, help='Similarity threshold (lower = more matches)')
def configure(threshold: float):
    """Configure system settings"""
    print(f"⚙️  Enhanced System Configuration")
    print("=" * 40)
    
    print(f"📊 Current Settings:")
    print(f"   Similarity threshold: {threshold}")
    print(f"   Lower values (0.6) = more matches")
    print(f"   Higher values (0.8) = fewer but more precise matches")
    
    # Update settings
    settings.SIMILARITY_THRESHOLD = threshold
    print(f"✅ Updated similarity threshold to {threshold}")
    
    print(f"\n💡 Tips for Better Results:")
    print(f"   • Use longer time windows (--days 14+) for more perspectives") 
    print(f"   • Try broader search queries first")
    print(f"   • Lower similarity threshold if getting too few matches")

if __name__ == "__main__":
    cli()