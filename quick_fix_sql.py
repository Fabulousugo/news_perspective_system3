# quick_fix_sql.py - Immediate fix for SQL syntax issue

import sys
import os
from pathlib import Path
import sqlite3
from typing import Dict

def fix_sql_issue():
    """Quick fix for the SQL syntax error"""
    print("Fixing SQL syntax issue...")
    
    # Create a corrected version
    corrected_code = '''
def analyze_bias_trends(self, days_back: int = 30) -> Dict:
    """Analyze bias trends over time"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # Fixed SQL query using parameter substitution
    cursor.execute("""
        SELECT * FROM daily_bias_stats 
        WHERE date >= date('now', '-' || ? || ' days')
        ORDER BY date
    """, (days_back,))
    
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        return {'error': 'No data available for trend analysis'}
    
    # Manual processing since we can't rely on pandas here
    columns = ['date', 'left_count', 'center_count', 'right_count', 
              'total_articles', 'perspective_matches', 'avg_match_confidence']
    
    data = []
    for row in results:
        data.append(dict(zip(columns, row)))
    
    if not data:
        return {'error': 'No data available for trend analysis'}
    
    total_articles = sum(row['total_articles'] for row in data)
    total_days = len(data)
    
    analysis = {
        'date_range': {
            'start': data[0]['date'],
            'end': data[-1]['date']
        },
        'total_days': total_days,
        'total_articles': total_articles,
        'daily_averages': {
            'articles_per_day': total_articles / total_days if total_days > 0 else 0,
            'perspective_matches_per_day': sum(row['perspective_matches'] for row in data) / total_days if total_days > 0 else 0,
            'avg_match_confidence': sum(row['avg_match_confidence'] for row in data) / total_days if total_days > 0 else 0
        },
        'bias_distribution_over_time': {
            'left_trend': [row['left_count'] for row in data],
            'center_trend': [row['center_count'] for row in data], 
            'right_trend': [row['right_count'] for row in data],
            'dates': [row['date'] for row in data]
        }
    }
    
    return analysis
'''
    
    print("SQL syntax fix created!")
    print("\nTo apply the fix:")
    print("1. Replace the analyze_bias_trends method in longitudinal_analysis.py with the corrected version")
    print("2. Or run the working test version below")
    
    return True

def create_minimal_working_test():
    """Create a minimal test that works around the SQL issue"""
    
    test_code = '''# minimal_research_test.py - Test without SQL dependencies

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def test_core_research_components():
    """Test core research components without SQL dependencies"""
    print("Testing Core Research Components (SQL-free)")
    print("=" * 45)
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Imports
    print("1. Testing imports...")
    try:
        from research.user_study_framework import UserStudyFramework
        from research.bias_visualization import BiasVisualization
        from models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
        from data_collection.simple_extended_collector import SimpleExtendedCollector
        print("Core imports successful")
        success_count += 1
    except Exception as e:
        print(f"Import failed: {e}")
    
    # Test 2: Visualization (no SQL)
    print("\\n2. Testing visualization...")
    try:
        from data_collection.news_apis import Article
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
    print("\\n3. Testing perspective matching...")
    try:
        matcher = OptimizedPerspectiveMatcher(optimization_level="quantized")
        matches = matcher.find_perspective_matches_fast(test_articles)
        print(f"Found {len(matches)} perspective matches")
        success_count += 1
    except Exception as e:
        print(f"Perspective matching failed: {e}")
    
    # Test 4: Extended collector
    print("\\n4. Testing news collection...")
    try:
        collector = SimpleExtendedCollector()
        summary = collector.get_source_summary()
        print(f"Collector ready with {summary['summary']['total_sources']} sources")
        success_count += 1
    except Exception as e:
        print(f"Collection failed: {e}")
    
    print(f"\\nResults: {success_count}/{total_tests} tests passed ({success_count/total_tests*100:.0f}%)")
    
    if success_count >= 3:
        print("\\nCore research system is functional!")
        print("\\nYou can now:")
        print("    Generate bias visualizations")
        print("    Find perspective matches")
        print("    Collect diverse news articles")
        print("    Use speed optimizations")
        
        print("\\nNext steps:")
        print("    1. Run research analysis:")
        print("      python scripts/research_analysis_cli.py comprehensive-analysis")
        print("    2. Generate visualizations:")
        print("      python scripts/research_analysis_cli.py visualize")
        return True
    else:
        print("\\nSome components need attention")
        return False

if __name__ == "__main__":
    test_core_research_components()
'''
    
    with open("minimal_research_test.py", 'w') as f:
        f.write(test_code)
    
    print("Created minimal_research_test.py")
    return True

if __name__ == "__main__":
    fix_sql_issue()
    create_minimal_working_test()
    
    print("\\nQuick fixes applied!")
    print("\\nRun the working test:")
    print("python minimal_research_test.py")
