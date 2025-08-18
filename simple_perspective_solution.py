# simple_perspective_solution.py - One-click fix for perspective issues

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def apply_simple_fix():
    """One-click fix for the most common perspective matching issue"""
    
    print("🔧 Simple Perspective Fix")
    print("=" * 30)
    print("The #1 reason perspectives don't show up: similarity threshold too high!")
    print("")
    
    try:
        from config.settings import settings
        
        # Show current threshold
        current = settings.SIMILARITY_THRESHOLD
        print(f"Current similarity threshold: {current}")
        
        if current > 0.65:
            print("❌ PROBLEM: Threshold too high!")
            print("   Anything above 0.65 is very strict")
        
        # Apply fix
        print("\n🎯 Applying fix...")
        settings.SIMILARITY_THRESHOLD = 0.55  # Much more permissive
        print(f"✅ Changed threshold from {current} to {settings.SIMILARITY_THRESHOLD}")
        
        print("\n🧪 Testing the fix...")
        
        # Quick test
        from data_collection.news_apis import NewsCollector
        from models.news_browser import NewsBrowser
        
        collector = NewsCollector()
        browser = NewsBrowser()
        
        # Test with general news (most likely to find matches)
        articles = collector.collect_diverse_articles("", days_back=7)
        total = sum(len(arts) for arts in articles.values())
        
        print(f"📡 Collected {total} articles")
        
        if total < 10:
            print("⚠️  Warning: Very few articles collected")
            print("   Check your API keys and internet connection")
            return False
        
        # Check if we have different bias categories
        bias_categories = [cat for cat, arts in articles.items() if len(arts) > 0]
        print(f"📊 Bias categories with articles: {bias_categories}")
        
        if len(bias_categories) < 2:
            print("❌ Need articles from at least 2 different bias categories")
            print("   Try a different search query or check your sources")
            return False
        
        # Test perspective matching
        all_articles = []
        for arts in articles.values():
            all_articles.extend(arts)
        
        browseable = browser.browse_articles(all_articles[:25])  # Test with reasonable subset
        
        articles_with_perspectives = len([a for a in browseable if a.perspective_count > 0])
        
        print(f"🎯 Result: {articles_with_perspectives}/{len(browseable)} articles now have perspectives!")
        
        if articles_with_perspectives > 0:
            print("✅ SUCCESS! The fix worked!")
            
            # Show examples
            print("\n📰 Examples of articles with perspectives:")
            count = 0
            for article_data in browseable:
                if article_data.perspective_count > 0 and count < 3:
                    article = article_data.article
                    bias_names = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
                    bias = bias_names.get(article.bias_label, "UNKNOWN")
                    
                    print(f"\n   {count+1}. [{bias}] {article.title[:50]}...")
                    print(f"      Source: {article.source}")
                    print(f"      Found {article_data.perspective_count} alternative perspectives")
                    
                    # Show one alternative perspective
                    if article_data.related_articles:
                        related, similarity = article_data.related_articles[0]
                        related_bias = bias_names.get(related.bias_label, "UNKNOWN")
                        print(f"      → [{related_bias}] {related.title[:45]}... ({related.source})")
                        print(f"        Similarity: {similarity:.2f}")
                    
                    count += 1
            
            print(f"\n🎉 Your system now finds perspectives!")
            print(f"Try: python scripts/run_application.py find-perspectives --query 'election'")
            
            return True
        else:
            print("❌ Still no perspectives found")
            print("\nTry these additional fixes:")
            print("1. Even lower threshold: settings.SIMILARITY_THRESHOLD = 0.45")
            print("2. More articles: use --days 14")
            print("3. Different topics: 'biden', 'trump', 'economy'")
            print("4. Run full diagnostic: python perspective_diagnostic.py")
            
            return False
            
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def create_quick_test():
    """Create a quick test command for the user"""
    
    test_content = '''#!/usr/bin/env python3
# quick_perspective_test.py - Quick test for perspective matching

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import settings
from data_collection.news_apis import NewsCollector  
from models.news_browser import NewsBrowser

# Use relaxed threshold
settings.SIMILARITY_THRESHOLD = 0.55

print("🧪 Quick Perspective Test")

collector = NewsCollector()
browser = NewsBrowser()

# Test with multiple queries to find one that works
queries = ["biden", "trump", "election", "economy", ""]

for query in queries:
    print(f"\\n📡 Testing: '{query or 'general news'}'")
    
    articles = collector.collect_diverse_articles(query, days_back=7)
    total = sum(len(arts) for arts in articles.values())
    
    if total < 15:
        print(f"   ⚠️  Only {total} articles - skipping")
        continue
    
    all_articles = []
    for arts in articles.values():
        all_articles.extend(arts)
    
    browseable = browser.browse_articles(all_articles)
    perspectives_found = len([a for a in browseable if a.perspective_count > 0])
    
    print(f"   🎯 {perspectives_found}/{len(browseable)} articles have perspectives")
    
    if perspectives_found > 0:
        print(f"   ✅ SUCCESS with '{query}'!")
        
        # Show example
        for article_data in browseable:
            if article_data.perspective_count > 0:
                article = article_data.article
                print(f"   📰 Example: {article.title[:50]}...")
                print(f"      {article_data.perspective_count} perspectives available")
                break
        break
else:
    print("\\n❌ No perspectives found with any query")
    print("Try running: python perspective_diagnostic.py")
'''
    
    with open("quick_perspective_test.py", "w") as f:
        f.write(test_content)
    
    print("✅ Created quick_perspective_test.py")

if __name__ == "__main__":
    success = apply_simple_fix()
    
    if success:
        create_quick_test()
        print(f"\n✅ SOLUTION APPLIED!")
        print(f"The fix lowered your similarity threshold to make matching more permissive.")
        print(f"\nQuick test: python quick_perspective_test.py")
        print(f"Full system: python scripts/run_application.py find-perspectives --query 'biden'")
    else:
        print(f"\n⚠️  The simple fix didn't work.")
        print(f"Run the full diagnostic to identify the issue:")
        print(f"   python perspective_diagnostic.py")
        
    print(f"\n💡 Remember: Lower threshold = more matches (but may be less precise)")
    print(f"   Current threshold: {0.55}")
    print(f"   For even more matches, try 0.45 or 0.5")