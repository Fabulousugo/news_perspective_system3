# perspective_fix.py - Fix the most common perspective matching issues

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def fix_perspective_matching():
    """Apply targeted fixes for perspective matching issues"""
    
    print("Fixing Perspective Matching Issues")
    print("=" * 50)
    
    try:
        from config.settings import settings
        
        # Fix 1: Lower the similarity threshold significantly
        print("Fix 1: Lowering similarity threshold")
        original_threshold = settings.SIMILARITY_THRESHOLD
        settings.SIMILARITY_THRESHOLD = 0.55  # Much lower for better matching
        print(f"  Changed from {original_threshold} to {settings.SIMILARITY_THRESHOLD}")
        
        # Fix 2: Create an improved news browser with better matching
        from models.news_browser import NewsBrowser
        from models.similarity_detector import SimilarityDetector
        import numpy as np
        from collections import defaultdict
        
        class ImprovedNewsBrowser(NewsBrowser):
            """Enhanced browser with better perspective matching"""
            
            def __init__(self):
                super().__init__()
                # Use even more relaxed thresholds
                self.loose_threshold = 0.5   # Very relaxed for better recall
                self.tight_threshold = 0.65  # Still reasonable quality
            
            def _build_perspective_map(self, articles):
                """Improved perspective mapping with better text matching"""
                perspective_map = defaultdict(list)
                
                # Group by bias more carefully
                bias_groups = defaultdict(list)
                for article in articles:
                    if article.bias_label is not None:
                        bias_groups[article.bias_label].append(article)
                
                print(f"  Bias groups: {[(bias, len(arts)) for bias, arts in bias_groups.items()]}")
                
                if len(bias_groups) < 2:
                    print("  Need at least 2 bias groups for matching")
                    return perspective_map
                
                # Compare articles across bias groups
                bias_list = list(bias_groups.keys())
                total_comparisons = 0
                total_matches = 0
                
                for i, bias1 in enumerate(bias_list):
                    for j, bias2 in enumerate(bias_list):
                        if i >= j:  # Avoid duplicates
                            continue
                        
                        articles1 = bias_groups[bias1]
                        articles2 = bias_groups[bias2]
                        
                        print(f"  Comparing {len(articles1)} vs {len(articles2)} articles")
                        
                        for article1 in articles1:
                            # Improved text preparation
                            text1 = self._prepare_improved_text(article1)
                            
                            candidate_texts = []
                            for article2 in articles2:
                                candidate_texts.append(self._prepare_improved_text(article2))
                            
                            if not candidate_texts:
                                continue
                            
                            total_comparisons += 1
                            
                            try:
                                similarities = self.similarity_detector.find_similar_articles(
                                    text1, candidate_texts, top_k=5  # Get more candidates
                                )
                                
                                for idx, similarity in similarities:
                                    if similarity >= self.loose_threshold:  # Use loose threshold
                                        related_article = articles2[idx]
                                        perspective_map[article1.url].append((related_article, similarity))
                                        perspective_map[related_article.url].append((article1, similarity))
                                        total_matches += 1
                                        
                                        # Debug output for first few matches
                                        if total_matches <= 3:
                                            print(f"  Match {total_matches}: {similarity:.3f}")
                                            print(f"      {article1.title[:40]}... ({article1.source})")
                                            print(f"      {related_article.title[:40]}... ({related_article.source})")
                                        
                            except Exception as e:
                                print(f"  Similarity calculation failed: {e}")
                                continue
                
                print(f"  Results: {total_matches} matches from {total_comparisons} comparisons")
                return perspective_map
            
            def _prepare_improved_text(self, article):
                """Better text preparation for similarity matching"""
                # Combine multiple text fields
                text_parts = []
                
                # Always include title (most important)
                if article.title:
                    text_parts.append(article.title)
                
                # Include description if available
                if article.description and len(article.description) > 20:
                    text_parts.append(article.description)
                
                # Include content preview if available and description is short
                if article.content and (not article.description or len(article.description) < 100):
                    # Take first 200 characters of content
                    content_preview = article.content[:200]
                    text_parts.append(content_preview)
                
                combined_text = ' '.join(text_parts)
                
                # Clean up the text
                # Remove extra whitespace
                cleaned_text = ' '.join(combined_text.split())
                
                # Remove very short text (likely to cause bad matches)
                if len(cleaned_text) < 30:
                    cleaned_text = article.title  # Fallback to just title
                
                return cleaned_text.strip()
        
        # Test the improved browser
        print("\nTesting improved perspective matching...")
        
        from data_collection.news_apis import NewsCollector
        collector = NewsCollector()
        
        # Try with a topic that should have coverage across sources
        test_queries = ["biden", "trump", "economy", "ukraine", "climate"]
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            
            articles = collector.collect_diverse_articles(query, days_back=10)
            total_articles = sum(len(arts) for arts in articles.values())
            
            if total_articles < 15:
                print(f"  Only {total_articles} articles - skipping")
                continue
            
            print(f"  Collected {total_articles} articles")
            
            # Flatten articles
            all_articles = []
            for bias_arts in articles.values():
                all_articles.extend(bias_arts)
            
            # Test with improved browser
            improved_browser = ImprovedNewsBrowser()
            browseable = improved_browser.browse_articles(all_articles)
            
            articles_with_perspectives = len([a for a in browseable if a.perspective_count > 0])
            
            print(f"  Results: {articles_with_perspectives}/{len(browseable)} articles have perspectives")
            
            if articles_with_perspectives > 0:
                max_perspectives = max(a.perspective_count for a in browseable)
                avg_perspectives = sum(a.perspective_count for a in browseable) / len(browseable)
                
                print(f"  Max perspectives: {max_perspectives}")
                print(f"  Average perspectives: {avg_perspectives:.1f}")
                
                # Show examples
                print(f"  Examples:")
                count = 0
                for article_data in browseable:
                    if article_data.perspective_count > 0 and count < 2:
                        article = article_data.article
                        print(f"      - {article.title[:45]}... ({article.source})")
                        print(f"        {article_data.perspective_count} perspectives available")
                        count += 1
                
                print(f"\nSUCCESS with query '{query}'!")
                print(f"  Run this: python scripts/run_application.py find-perspectives --query '{query}' --days 10")
                break
        else:
            print(f"\nNo good matches found. Try:")
            print(f"  1. General news (no query)")
            print(f"  2. Breaking news topics")
            print(f"  3. Increase days_back to 14+")
        
        return True
        
    except Exception as e:
        print(f"Fix failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def create_perspective_test_script():
    """Create a simple test script for perspective matching"""
    
    test_script = '''# test_perspectives.py - Simple perspective matching test

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
'''
    
    with open("test_perspectives.py", "w") as f:
        f.write(test_script)
    
    print("Created test_perspectives.py")

if __name__ == "__main__":
    print("Perspective Matching Fix Tool")
    print("=" * 50)
    
    # Apply fixes
    if fix_perspective_matching():
        # Create test script
        create_perspective_test_script()
        
        print(f"\nFixes Applied Successfully!")
        print(f"\nNext steps:")
        print(f"1. Test the fixes: python test_perspectives.py")
        print(f"2. Try with your system: python scripts/run_application.py find-perspectives --query 'biden' --days 10")
        print(f"3. If still no matches, run: python perspective_diagnostic.py")
        
        print(f"\nKey changes made:")
        print(f"  - Lowered similarity threshold to 0.55")
        print(f"  - Improved text preparation for matching")
        print(f"  - Better cross-bias comparison")
        print(f"  - More relaxed matching criteria")
    else:
        print(f"\nFix failed - try the diagnostic tool:")
        print(f"  python perspective_diagnostic.py")