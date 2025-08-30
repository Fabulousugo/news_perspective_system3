# test_core_matching.py
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
from src.data_collection.simple_extended_collector import SimpleExtendedCollector
from config import settings

# Set threshold
settings.SIMILARITY_THRESHOLD = 0.40

collector = SimpleExtendedCollector()
matcher = OptimizedPerspectiveMatcher()
matcher.similarity_detector.threshold = 0.40

print(f"Testing with threshold: {matcher.similarity_detector.threshold}")

# Collect articles
articles_dict = collector.collect_diverse_articles("election", days_back=5)
all_articles = []
for articles in articles_dict.values():
    all_articles.extend(articles)

print(f"Collected {len(all_articles)} articles")

# Show sample titles to verify topic overlap
print("\nSample article titles:")
for i, article in enumerate(all_articles[:10]):
    print(f"{i+1}. [{article.source}] {article.title}")

# Test matching
matches = matcher.find_perspective_matches_fast(all_articles[:30])
print(f"\nMatches found: {len(matches)}")

if matches:
    for i, match in enumerate(matches[:3]):
        print(f"\nMatch {i+1}: {match.topic} (confidence: {match.confidence:.3f})")
        for bias, article in match.articles.items():
            print(f"  {bias}: {article.title[:50]}... ({article.source})")
else:
    print("No matches found - system is broken")