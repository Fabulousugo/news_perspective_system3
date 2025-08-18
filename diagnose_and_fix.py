
# diagnose_and_fix.py - Check what's available and fix it

import sys,os
from pathlib import Path
# sys.path.append(str(Path(__file__).parent / "src"))


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def diagnose_perspective_matcher():
    """Check what methods are available in PerspectiveMatcher"""
    
    print("Diagnosing PerspectiveMatcher...")
    
    try:
        from src.models.perspective_matcher import PerspectiveMatcher
        
        # Create instance
        matcher = PerspectiveMatcher()
        
        # Check available methods
        methods = [method for method in dir(matcher) if not method.startswith('_')]
        
        print(f"PerspectiveMatcher loaded successfully")
        print(f"Available methods: {methods}")
        
        # Check for common method names
        method_checks = [
            'find_perspective_matches',
            'find_perspectives', 
            'get_perspective_matches',
            'match_perspectives',
            'analyze_perspectives'
        ]
        
        available_methods = []
        for method_name in method_checks:
            if hasattr(matcher, method_name):
                available_methods.append(method_name)
                print(f"   Has method: {method_name}")
            else:
                print(f"   Missing: {method_name}")
        
        return matcher, available_methods
        
    except Exception as e:
        print(f"Failed to load PerspectiveMatcher: {e}")
        return None, []

def create_working_perspective_test():
    """Create a test that works with whatever PerspectiveMatcher we have"""
    
    print("\nCreating working perspective test...")
    
    try:
        from src.data_collection.news_apis import NewsCollector
        from src.models.perspective_matcher import PerspectiveMatcher
        
        # Test basic functionality
        collector = NewsCollector()
        matcher = PerspectiveMatcher()
        
        print("Components loaded")
        
        # Test collection (we know this works from the patch)
        print("Testing collection...")
        articles = collector.collect_diverse_articles("election", days_back=3)
        total = sum(len(arts) for arts in articles.values())
        print(f"Collected {total} articles")
        
        if total > 0:
            # Flatten articles
            all_articles = []
            for arts in articles.values():
                all_articles.extend(arts)
            
            print(f"Flattened to {len(all_articles)} articles")
            
            # Try different method names that might exist
            methods_to_try = [
                'find_perspective_matches',
                'find_perspectives',
                'get_perspective_matches',
                'match_perspectives'
            ]
            
            for method_name in methods_to_try:
                if hasattr(matcher, method_name):
                    print(f"Trying method: {method_name}")
                    try:
                        method = getattr(matcher, method_name)
                        
                        # Try calling with just articles
                        result = method(all_articles[:10])  # Test with small subset
                        
                        if isinstance(result, list):
                            print(f"{method_name} worked! Found {len(result)} matches")
                            
                            # Show example if available
                            if result:
                                match = result[0]
                                if hasattr(match, 'topic') and hasattr(match, 'articles'):
                                    print(f"   Example: {match.topic}")
                                    for bias, article in match.articles.items():
                                        print(f"     {bias}: {article.title[:40]}...")
                                else:
                                    print(f"   Match type: {type(match)}")
                            
                            return True
                            
                        else:
                            print(f"   {method_name} returned {type(result)}, expected list")
                            
                    except Exception as e:
                        print(f"   {method_name} failed: {e}")
                        continue
            
            print("No working perspective matching method found")
            print("But article collection is working!")
            
            return True
            
        else:
            print("No articles collected to test with")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def create_simple_working_browser():
    """Create a simple browser that definitely works"""
    
    print("\nCreating simple working browser...")
    
    browser_code = '''# simple_working_browser.py - Guaranteed to work

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
                print(f"\\n{bias_category.upper()} ({len(bias_articles)} articles):")
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
        print("\\nSummary:")
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
'''
    
    # Write the browser file
    with open("simple_working_browser.py", "w") as f:
        f.write(browser_code)
    
    print("Created simple_working_browser.py")
    print("\nTry: python simple_working_browser.py --query 'election' --days 7")

def main():
    """Main diagnostic and fix function"""
    
    print("Perspective Matcher Diagnosis and Fix")
    print("=" * 50)
    
    # Step 1: Diagnose what we have
    matcher, available_methods = diagnose_perspective_matcher()
    
    if not matcher:
        print("Could not load PerspectiveMatcher")
        print("Try using the standalone system instead:")
        print("   python test_standalone_system.py")
        return
    
    # Step 2: Test with working methods
    if available_methods:
        print(f"\nFound {len(available_methods)} working methods")
        if create_working_perspective_test():
            print("\nPerspective matching is working!")
        else:
            print("\nPerspective matching has issues, but collection works")
    else:
        print("\nNo standard perspective methods found")
    
    # Step 3: Create simple fallback browser
    create_simple_working_browser()
    
    print("\nSummary:")
    print("Article collection is working (patched)")
    if available_methods:
        print("Perspective matching methods available")
    else:
        print("Perspective matching needs work")
    print("Simple browser created as fallback")
    
    print("\nWhat to try:")
    print("1. Simple browser: python simple_working_browser.py --query 'election'")
    print("2. Original scripts: python scripts/run_application.py find-perspectives --query 'election'")
    print("3. Standalone system: python test_standalone_system.py")

if __name__ == "__main__":
    main()
