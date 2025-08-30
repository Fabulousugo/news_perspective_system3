# scripts/optimized_test.py

import os,sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.simple_extended_collector import SimpleExtendedCollector
from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher

collector = SimpleExtendedCollector()
matcher = OptimizedPerspectiveMatcher()

# Lower threshold for better recall
matcher.similarity_detector.threshold = 0.42

# Get more articles
articles_dict = collector.collect_diverse_articles(days_back=5, query="")
all_articles = []
for bias_arts in articles_dict.values():
    all_articles.extend(bias_arts[:20])  # 20 from each category

print(f"Testing with {len(all_articles)} articles...")
matches = matcher.find_perspective_matches_fast(all_articles)

print(f"Found {len(matches)} perspective matches!")

for i, match in enumerate(matches[:10]):  # Show first 10
    print(f"\n#{i+1}: {match.topic} (confidence: {match.confidence:.3f})")
    for bias, article in match.articles.items():
        print(f"  {bias}: {article.source} - {article.title[:70]}...")