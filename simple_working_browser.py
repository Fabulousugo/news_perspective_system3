# simple_working_browser.py - Guaranteed to work

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def simple_browse(query="", days=7):
    """Simple news browser that just shows articles by bias"""
    
    print(f"Simple News Browser")
    print("=" * 40)
    print(f"Query: {query or 'General news'}")
    print(f"Days: {days}")
    print()
    
    try:
        from data_collection.news_apis import NewsCollector
        
        # Collect articles
        collector = NewsCollector()
        articles = collector.collect_diverse_articles(query, days)
        
        total = sum(len(arts) for arts in articles.values())
        print(f"Collected {total} articles")
        
        if total == 0:
            print("No articles found")
            return
        
        # Show articles by bias
        bias_names = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
        
        for bias_category, bias_articles in articles.items():
            if bias_articles:
                print(f"\n{bias_category.upper()} ({len(bias_articles)} articles):")
                print("=" * 50)
                
                for i, article in enumerate(bias_articles[:10]):  # Show first 10
                    bias_icon = bias_names.get(article.bias_label, "?")
                    time_str = article.published_at.strftime("%m/%d %H:%M")
                    
                    print(f"[{i+1:2d}] {bias_icon} | {time_str}")
                    print(f"     {article.title}")
                    print(f"     {article.source}")
                    print(f"     {article.url}")
                    print()
        
        # Show summary
        print("\nSummary:")
        for bias_category, bias_articles in articles.items():
            if bias_articles:
                sources = list(set(a.source for a in bias_articles))
                print(f"   {bias_category}: {len(bias_articles)} articles from {len(sources)} sources")
                print(f"       Sources: {', '.join(sources)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', default='', help='Search query')
    parser.add_argument('--days', '-d', type=int, default=7, help='Days back')
    args = parser.parse_args()
    
    simple_browse(args.query, args.days)
