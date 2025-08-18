# scripts/improved_news_browser.py - Better user interface

import click
import logging
from pathlib import Path
import sys,os
from datetime import datetime
import webbrowser

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


from src.data_collection.news_apis import NewsCollector
from src.models.news_browser import NewsBrowser, ArticleWithPerspectives
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """📰 News Perspective Browser - Discover diverse viewpoints on news stories"""
    pass

@cli.command()
@click.option('--query', '-q', default='', help='Search query (leave empty for general news)')
@click.option('--days', '-d', default=7, help='Days to look back (more days = more perspectives)')
@click.option('--sort', '-s', type=click.Choice(['recent', 'diverse']), default='diverse', 
              help='Sort by: recent (newest first) or diverse (most perspectives first)')
@click.option('--limit', '-l', default=20, help='Number of articles to show')
def browse(query: str, days: int, sort: str, limit: int):
    """Browse news articles and see available perspectives"""
    
    print(f"📰 News Perspective Browser")
    print("=" * 60)
    print(f"🔍 Query: {query or 'General news'}")
    print(f"📅 Looking back: {days} days")
    print(f"🔄 Sort by: {sort}")
    print("")
    
    try:
        # Initialize components
        collector = NewsCollector()
        browser = NewsBrowser()
        
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
        
        # Browse articles
        print("🔍 Analyzing for perspectives...")
        browseable_articles = browser.browse_articles(all_articles, sort_by=sort)
        
        # Show statistics
        stats = browser.get_statistics(browseable_articles)
        print(f"\n📊 Statistics:")
        print(f"   📰 Total articles: {stats['total_articles']}")
        print(f"   🎯 Articles with perspectives: {stats['articles_with_perspectives']}")
        print(f"   📈 Perspective coverage: {stats['perspective_coverage']:.1%}")
        print(f"   🔗 Average perspectives per article: {stats['average_perspectives_per_article']:.1f}")
        print(f"   🏆 Max perspectives found: {stats['max_perspectives_found']}")
        
        print(f"\n🏛️  Bias distribution:")
        for bias, count in stats['bias_distribution'].items():
            print(f"   {bias}: {count} articles")
        
        # Show articles
        print(f"\n📋 Articles (showing top {limit}):")
        print("=" * 100)
        
        for i, browseable in enumerate(browseable_articles[:limit]):
            article = browseable.article
            
            # Format publish time
            time_str = article.published_at.strftime("%m/%d %H:%M")
            
            # Bias indicator
            bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
            bias_indicator = bias_names.get(article.bias_label, "❓ UNKNOWN")
            
            # Perspective indicator
            if browseable.perspective_count > 0:
                perspective_indicator = f"🎯 {browseable.perspective_count} perspectives"
            else:
                perspective_indicator = "🔍 No perspectives found"
            
            print(f"\n[{i+1:2d}] {bias_indicator} | {time_str} | {perspective_indicator}")
            print(f"     📰 {article.title}")
            print(f"     🏢 {article.source}")
            
            # Show perspectives if available
            if browseable.perspective_count > 0:
                print(f"     🔗 Alternative perspectives:")
                for j, (related_article, similarity) in enumerate(browseable.related_articles[:3]):
                    related_bias = bias_names.get(related_article.bias_label, "❓")
                    print(f"        {related_bias} ({similarity:.2f}) {related_article.title[:60]}...")
                    print(f"           📰 {related_article.source}")
            
            print("     " + "─" * 90)
        
        # Interactive menu
        if browseable_articles:
            print(f"\n🎯 Interactive Options:")
            print(f"   • Enter article number (1-{min(limit, len(browseable_articles))}) to read full article")
            print(f"   • Type 'perspectives X' to see all perspectives for article X")
            print(f"   • Type 'search KEYWORD' to filter articles")
            print(f"   • Type 'exit' to quit")
            
            while True:
                try:
                    user_input = input("\n> ").strip().lower()
                    
                    if user_input == 'exit':
                        break
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
                        print("❌ Unknown command. Try: article number, 'perspectives X', 'search KEYWORD', or 'exit'")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Browse command failed: {e}")

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
        # Show first 500 characters
        content_preview = article.content[:500]
        if len(article.content) > 500:
            content_preview += "... [content truncated]"
        print(f"{content_preview}")
    
    print(f"\n🎯 Perspectives Available: {browseable.perspective_count}")
    
    if browseable.perspective_count > 0:
        print(f"Alternative viewpoints:")
        for i, (related_article, similarity) in enumerate(browseable.related_articles):
            bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
            bias_indicator = bias_names.get(related_article.bias_label, "❓")
            print(f"  {i+1}. {bias_indicator} ({similarity:.2f}) {related_article.title}")
            print(f"     📰 {related_article.source}")
    
    print("=" * 100)
    
    # Ask if user wants to open in browser
    open_browser = input("\n🌐 Open original article in browser? (y/n): ").strip().lower()
    if open_browser == 'y':
        webbrowser.open(article.url)

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
    
    for i, browseable in enumerate(matching_articles[:10]):  # Show top 10
        article = browseable.article
        bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
        bias_indicator = bias_names.get(article.bias_label, "❓")
        
        print(f"[{i+1}] {bias_indicator} | 🎯 {browseable.perspective_count} perspectives")
        print(f"    📰 {article.title}")
        print(f"    🏢 {article.source}")
        print()

@cli.command()
@click.option('--threshold', '-t', default=0.65, help='Similarity threshold (lower = more matches)')
def configure(threshold: float):
    """Configure system settings"""
    print(f"⚙️  Current Configuration:")
    print(f"   Similarity threshold: {threshold}")
    print(f"   This determines how similar articles need to be to count as 'same story'")
    print(f"   Lower values (0.6) = more matches, higher values (0.8) = fewer but more precise")
    
    # Update settings
    settings.SIMILARITY_THRESHOLD = threshold
    print(f"✅ Updated similarity threshold to {threshold}")

if __name__ == "__main__":
    cli()