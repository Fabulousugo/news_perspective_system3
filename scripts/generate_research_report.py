# scripts/generate_research_report.py
"""
Dissertation Research Report Generator
Runs the full pipeline and outputs automated findings.
"""

import time
import json
from datetime import datetime
from pathlib import Path
import os,sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import your system components
from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
from src.data_collection.enhanced_news_collector import EnhancedNewsCollector
from config.settings import settings

def main():
    print("📊 Generating Automated Research Report\n")
    print("=" * 60)
    
    # Initialize components
    print("🔧 Initializing system components...")
    collector = EnhancedNewsCollector(settings)
    matcher = OptimizedPerspectiveMatcher(optimization_level="quantized", use_source_bias=True)
    
    # Step 1: Collect real articles
    print("📥 Collecting diverse news articles...")
    start_time = time.time()
    
    # Collect from all sources (last 24 hours)
    articles = collector.collect_diverse_articles(
        query="",           
        days_back=1,        
        strategy="comprehensive"  # ✅ Correct parameter name
    )
    
    # Flatten articles from all bias groups
    all_articles = []
    for bias_group, bias_articles in articles.items():
        print(f"   {bias_group.title()}: {len(bias_articles)} articles")
        all_articles.extend(bias_articles)
    
    collection_time = time.time() - start_time
    print(f"✅ Collection complete in {collection_time:.2f}s\n")
    
    if not all_articles:
        print("❌ No articles collected. Check your API keys and network connection.")
        return
    
    # Step 2: Find perspective matches
    print("🔍 Finding cross-perspective matches...")
    match_start = time.time()
    matches = matcher.find_perspective_matches_fast(all_articles)
    match_time = time.time() - match_start
    print(f"✅ Found {len(matches)} matches in {match_time:.2f}s\n")
    
    if not matches:
        print("❌ No matches found. Try with more articles or lower similarity threshold.")
        return
    
    # Step 3: Generate automated research report
    print("🧠 Generating research report...")
    report = matcher.generate_research_report(matches)
    
    # Step 4: Print findings (this answers your RQs)
    print("=" * 60)
    print("🎯 DISSERTATION RESEARCH FINDINGS")
    print("=" * 60)
    print(report['findings'])
    
    # Optional: Save full report to file
    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_report = {
        "timestamp": datetime.now().isoformat(),
        "collection_stats": {
            "total_articles": len(all_articles),
            "collection_time_seconds": collection_time
        },
        "matching_stats": {
            "total_matches": len(matches),
            "matching_time_seconds": match_time,
            "average_confidence": sum(m.confidence for m in matches) / len(matches)
        },
        **report  # Includes findings, polarities, etc.
    }
    
    report_path = output_dir / f"research_report_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print(f"\n📄 Full report saved to: {report_path}")
    print("✅ You can now copy the findings into your dissertation.")

if __name__ == "__main__":
    main()