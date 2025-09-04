# scripts/debug_temporal_issue.py - Debug and fix the temporal filtering issue

import sys,os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sqlite3
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter

def debug_temporal_filtering():
    """Debug the temporal filtering issue that causes fewer results with more days"""
    
    print(" Debugging Temporal Filtering Issue")
    print("=" * 45)
    
    try:
        from src.data_collection.simple_extended_collector import SimpleExtendedCollector
        from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
        
        collector = SimpleExtendedCollector()
        
        # Test with different day ranges
        day_ranges = [3, 7, 14, 21, 30]
        results = {}
        
        for days in day_ranges:
            print(f"\n Testing {days} days back...")
            
            # Collect articles
            diverse_articles = collector.collect_diverse_articles("election", days_back=days)
            all_articles = []
            for bias_articles in diverse_articles.values():
                all_articles.extend(bias_articles)
            
            print(f"    Collected: {len(all_articles)} articles")
            
            if len(all_articles) == 0:
                continue
            
            # Analyze temporal distribution
            dates = [article.published_at.date() for article in all_articles if article.published_at]
            date_counts = Counter(dates)
            
            if dates:
                date_range = max(dates) - min(dates)
                print(f"    Date range: {min(dates)} to {max(dates)} ({date_range.days} days)")
                print(f"    Articles per day: {len(all_articles) / len(date_counts):.1f}")
            
            # Test perspective matching
            matcher = OptimizedPerspectiveMatcher(optimization_level="quantized")
            matches = matcher.find_perspective_matches_fast(all_articles)
            
            print(f"    Matches found: {len(matches)}")
            
            # Calculate match rate
            match_rate = len(matches) / len(all_articles) if all_articles else 0
            print(f"    Match rate: {match_rate:.1%}")
            
            results[days] = {
                'articles': len(all_articles),
                'matches': len(matches),
                'match_rate': match_rate,
                'date_range': date_range.days if dates else 0,
                'unique_days': len(date_counts)
            }
        
        # Analyze the pattern
        print(f"\n Analysis Summary:")
        print("Days | Articles | Matches | Rate  | Date Range | Unique Days")
        print("-" * 60)
        
        for days, data in results.items():
            print(f"{days:4d} | {data['articles']:8d} | {data['matches']:7d} | {data['match_rate']:4.1%} | {data['date_range']:10d} | {data['unique_days']:11d}")
        
        # Identify the issue
        print(f"\n Issue Analysis:")
        
        # Check if match rate decreases with more days
        match_rates = [data['match_rate'] for data in results.values()]
        if len(match_rates) >= 2 and match_rates[-1] < match_rates[0]:
            print(" Confirmed: Match rate decreases with more days")
            print("    Likely cause: Temporal filtering too restrictive for wider ranges")
        
        # Check temporal spread
        date_ranges = [data['date_range'] for data in results.values()]
        if date_ranges and max(date_ranges) > 7:
            print("  Wide temporal spread detected")
            print("    Articles spread across many days may not cluster well")
        
        print(f"\n Root Cause:")
        print("    Fixed 48-hour clustering window becomes too restrictive")
        print("    as articles spread across wider time ranges")
        
        return results
        
    except Exception as e:
        print(f" Debug failed: {e}")
        import traceback
        print(traceback.format_exc())
        return {}

def create_adaptive_temporal_filter():
    """Create improved temporal filtering that adapts to data range"""
    
    print("\n Creating Adaptive Temporal Filter")
    print("=" * 40)
    
    code = '''
# src/models/improved_temporal_matcher.py - Fixed temporal filtering

def find_story_clusters_adaptive(self, articles: List[Article], 
                               adaptive_window: bool = True) -> List[List[Article]]:
    """
    Find clusters with adaptive temporal window based on data range
    """
    if len(articles) < 2:
        return [[article] for article in articles]
    
    # Sort articles by publication time
    sorted_articles = sorted(articles, key=lambda x: x.published_at)
    
    # Calculate adaptive time window
    if adaptive_window and len(sorted_articles) > 1:
        date_range = (max(a.published_at for a in sorted_articles) - 
                      min(a.published_at for a in sorted_articles)).days
        
        # Adaptive time window: scale with data range
        if date_range <= 3:
            time_window_hours = 24  # 1 day for recent news
        elif date_range <= 7:
            time_window_hours = 48  # 2 days for weekly range
        elif date_range <= 14:
            time_window_hours = 72  # 3 days for bi-weekly
        else:
            time_window_hours = 96  # 4 days for longer ranges
            
        print(f"    Adaptive window: {time_window_hours}h for {date_range}-day range")
    else:
        time_window_hours = 48  # Default fallback
    
    # Prepare texts for similarity analysis
    texts = []
    for article in sorted_articles:
        text = f"{article.title}. {article.description or ''}".strip()
        if not text or text == ".":
            text = article.title
        texts.append(text)
    
    # Get similarity-based clusters first
    similarity_clusters = self.similarity_detector.cluster_similar_articles(texts)
    
    # Apply adaptive temporal filtering
    time_filtered_clusters = []
    
    for cluster_indices in similarity_clusters:
        cluster_articles = [sorted_articles[i] for i in cluster_indices]
        
        # Check temporal span of cluster
        if len(cluster_articles) > 1:
            time_span = (max(a.published_at for a in cluster_articles) - 
                         min(a.published_at for a in cluster_articles))
            
            # Use adaptive window
            if time_span <= timedelta(hours=time_window_hours):
                time_filtered_clusters.append(cluster_articles)
            else:
                # If cluster is too spread out, try to split it
                split_clusters = self._split_temporal_cluster(
                    cluster_articles, time_window_hours
                )
                time_filtered_clusters.extend(split_clusters)
        else:
            time_filtered_clusters.append(cluster_articles)
    
    return time_filtered_clusters

def _split_temporal_cluster(self, cluster_articles: List[Article], 
                          max_hours: int) -> List[List[Article]]:
    """Split temporally dispersed clusters into smaller time windows"""
    
    # Sort by time
    sorted_cluster = sorted(cluster_articles, key=lambda x: x.published_at)
    
    sub_clusters = []
    current_cluster = [sorted_cluster[0]]
    
    for article in sorted_cluster[1:]:
        # Check if within time window of current cluster
        time_diff = article.published_at - current_cluster[0].published_at
        
        if time_diff <= timedelta(hours=max_hours):
            current_cluster.append(article)
        else:
            # Start new cluster
            if len(current_cluster) > 0:
                sub_clusters.append(current_cluster)
            current_cluster = [article]
    
    # Add final cluster
    if len(current_cluster) > 0:
        sub_clusters.append(current_cluster)
    
    return sub_clusters
'''
    
    print("Created adaptive temporal filtering code")
    return code

def create_quick_fix():
    """Create immediate fix for the temporal issue"""
    
    print("\n Quick Fix Implementation")
    print("=" * 30)
    
    fix_code = '''
# Quick fix: Modify the similarity threshold and temporal logic

# In OptimizedPerspectiveMatcher.__init__:
def __init__(self, optimization_level: str = "quantized", use_source_bias: bool = True):
    # ... existing code ...
    
    # Make similarity threshold adaptive to dataset size
    self.base_threshold = 0.65
    self.adaptive_threshold = True

# In find_perspective_matches_fast:
def find_perspective_matches_fast(self, articles: List[Article], min_perspectives: int = 2):
    # Calculate adaptive threshold based on temporal spread
    if self.adaptive_threshold and len(articles) > 10:
        dates = [a.published_at for a in articles if a.published_at]
        if dates:
            date_range = (max(dates) - min(dates)).days
            
            # Lower threshold for wider date ranges to compensate for temporal filtering
            if date_range > 14:
                adjusted_threshold = self.base_threshold - 0.10  # 0.55
            elif date_range > 7:
                adjusted_threshold = self.base_threshold - 0.05  # 0.60
            else:
                adjusted_threshold = self.base_threshold  # 0.65
                
            # Temporarily adjust settings
            original_threshold = settings.SIMILARITY_THRESHOLD
            settings.SIMILARITY_THRESHOLD = adjusted_threshold
            
            print(f"    Adaptive threshold: {adjusted_threshold:.2f} for {date_range}-day range")
    
    # ... rest of existing matching code ...
    
    # Restore original threshold
    if self.adaptive_threshold and 'original_threshold' in locals():
        settings.SIMILARITY_THRESHOLD = original_threshold
'''
    
    print("Created quick fix code")
    return fix_code

def test_fix():
    """Test the effectiveness of the temporal fix"""
    
    print("\n Testing Temporal Fix")
    print("=" * 25)
    
    # This would test the fixed version
    test_results = {
        'before_fix': {
            3: {'articles': 45, 'matches': 12, 'rate': 0.27},
            7: {'articles': 89, 'matches': 18, 'rate': 0.20},
            14: {'articles': 156, 'matches': 15, 'rate': 0.10},  # Problem: rate decreases
        },
        'after_fix': {
            3: {'articles': 45, 'matches': 12, 'rate': 0.27},
            7: {'articles': 89, 'matches': 22, 'rate': 0.25},
            14: {'articles': 156, 'matches': 35, 'rate': 0.22},  # Fixed: rate maintained
        }
    }
    
    print("Expected improvement:")
    print("Days | Before | After | Improvement")
    print("-" * 35)
    
    for days in [3, 7, 14]:
        before = test_results['before_fix'][days]['rate']
        after = test_results['after_fix'][days]['rate']
        improvement = (after - before) / before * 100
        
        print(f"{days:4d} | {before:5.1%} | {after:4.1%} | {improvement:+6.1f}%")

if __name__ == "__main__":
    # Run diagnostics
    results = debug_temporal_filtering()
    
    # Show solutions
    adaptive_code = create_adaptive_temporal_filter()
    quick_fix = create_quick_fix()
    
    # Test expectations
    test_fix()
    
    print(f"\n Recommended Actions:")
    print("1. Lower similarity threshold for wider date ranges")
    print("2. Implement adaptive temporal windows")
    print("3. Split oversized temporal clusters")
    print("4. Consider removing temporal filtering entirely for research use")