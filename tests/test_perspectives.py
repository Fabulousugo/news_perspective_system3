# test_perspectives.py - Simple perspective matching test

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def test_perspectives_simple():
    from data_collection.news_apis import NewsCollector
    from models.news_browser import NewsBrowser
    from config.settings import settings
    
    # Use very relaxed threshold
    settings.SIMILARITY_THRESHOLD = 0.55
    
    print("Simple Perspective Test")
    print("=" * 40)
    
    collector = NewsCollector()
    browser = NewsBrowser()
    
    # Test with general news (no query) - most likely to find matches
    print("Collecting general news...")
    articles = collector.collect_diverse_articles("", days_back=10)
    
    total = sum(len(arts) for arts in articles.values())
    print(f"Collected {total} articles")
    
    if total < 10:
        print("Too few articles - check API keys")
        return
    
    # Check bias distribution
    bias_counts = {}
    for bias_cat, arts in articles.items():
        if arts:
            bias_counts[bias_cat] = len(arts)
    
    print(f"Bias distribution: {bias_counts}")
    
    if len(bias_counts) < 2:
        print("Need articles from at least 2 different bias categories")
        return
    
    # Flatten and test
    all_articles = []
    for arts in articles.values():
        all_articles.extend(arts)
    
    print("Testing perspective matching...")
    browseable = browser.browse_articles(all_articles[:30])  # Test with subset
    
    with_perspectives = [a for a in browseable if a.perspective_count > 0]
    
    print(f"Result: {len(with_perspectives)}/{len(browseable)} articles have perspectives")
    
    if with_perspectives:
        print("SUCCESS! Examples:")
        for i, article_data in enumerate(with_perspectives[:3]):
            article = article_data.article
            print(f"  {i+1}. {article.title[:50]}... ({article.source})")
            print(f"     {article_data.perspective_count} alternative perspectives")
    else:
        print("No perspectives found")
        print("Try:")
        print("  - Lower threshold: settings.SIMILARITY_THRESHOLD = 0.5")
        print("  - More articles: days_back=14")
        print("  - Different topic: query='election' or 'biden'")

if __name__ == "__main__":
    test_perspectives_simple()
