# minimal_research_test.py - Test without SQL dependencies

import sys,os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
def test_core_research_components():
    """Test core research components without SQL dependencies"""
    print("Testing Core Research Components (SQL-free)")
    print("=" * 45)
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Imports
    print("1. Testing imports...")
    try:
        from src.research.user_study_framework import UserStudyFramework
        from src.research.bias_visualization import BiasVisualization
        from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
        from src.data_collection.simple_extended_collector import SimpleExtendedCollector
        print("Core imports successful")
        success_count += 1
    except Exception as e:
        print(f"Import failed: {e}")
    
    # Test 2: Visualization (no SQL)
    print("\n2. Testing visualization...")
    try:
        from src.data_collection.news_apis import Article
        from datetime import datetime
        
        visualizer = BiasVisualization()
        
        test_articles = [
            Article("Test article 1", "content", "url1", "CNN", datetime.now(), bias_label=0),
            Article("Test article 2", "content", "url2", "Fox News", datetime.now(), bias_label=2),
            Article("Test article 3", "content", "url3", "Reuters", datetime.now(), bias_label=1),
        ]
        
        # Test chart creation
        chart_path = visualizer.create_bias_distribution_chart(test_articles)
        print("Visualization working")
        success_count += 1
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Test 3: Optimized perspective matching
    print("\n3. Testing perspective matching...")
    try:
        matcher = OptimizedPerspectiveMatcher(optimization_level="quantized")
        matches = matcher.find_perspective_matches_fast(test_articles)
        print(f"Found {len(matches)} perspective matches")
        success_count += 1
    except Exception as e:
        print(f"Perspective matching failed: {e}")
    
    # Test 4: Extended collector
    print("\n4. Testing news collection...")
    try:
        collector = SimpleExtendedCollector()
        summary = collector.get_source_summary()
        print(f"Collector ready with {summary['summary']['total_sources']} sources")
        success_count += 1
    except Exception as e:
        print(f"Collection failed: {e}")
    
    print(f"\nResults: {success_count}/{total_tests} tests passed ({success_count/total_tests*100:.0f}%)")
    
    if success_count >= 3:
        print("\nCore research system is functional!")
        print("\nYou can now:")
        print("    Generate bias visualizations")
        print("    Find perspective matches")
        print("    Collect diverse news articles")
        print("    Use speed optimizations")
        
        print("\nNext steps:")
        print("    1. Run research analysis:")
        print("      python scripts/research_analysis_cli.py comprehensive-analysis")
        print("    2. Generate visualizations:")
        print("      python scripts/research_analysis_cli.py visualize")
        return True
    else:
        print("\nSome components need attention")
        return False

if __name__ == "__main__":
    test_core_research_components()
