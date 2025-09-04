# fix_temporal_issue.py - Immediate fix for the temporal filtering problem

import sys
from pathlib import Path

def apply_immediate_fix():
    """Apply immediate fix to resolve the temporal filtering issue"""
    
    print("Applying Immediate Temporal Fix")
    print("=" * 35)
    
    # The core issue is in the temporal filtering logic
    # Here's the immediate fix:
    
    fix_instructions = """
PROBLEM IDENTIFIED:
The temporal filtering uses a fixed 48-hour window that becomes too restrictive 
when articles are spread across longer time periods.

IMMEDIATE FIX:
In your OptimizedPerspectiveMatcher class, modify the similarity threshold
and temporal logic to be adaptive.

SOLUTION:
"""
    
    print(fix_instructions)
    
    # Create the fix
    print("1. Create this fixed version:")
    
    quick_fix_code = '''
# In src/models/optimized_perspective_matcher.py
# Replace the _find_cross_perspective_matches_fast method with this:

def _find_cross_perspective_matches_fast(self, bias_groups: Dict[str, List[Article]]) -> List[PerspectiveMatch]:
    """Fixed cross-perspective matching without restrictive temporal filtering"""
    matches = []
    bias_list = list(bias_groups.keys())
    
    # ADAPTIVE THRESHOLD: Lower threshold for wider date ranges
    all_articles_for_analysis = []
    for articles in bias_groups.values():
        all_articles_for_analysis.extend(articles)
    
    dates = [a.published_at for a in all_articles_for_analysis if a.published_at]
    if dates:
        date_range = (max(dates) - min(dates)).days
        
        # Adaptive similarity threshold
        if date_range > 14:
            similarity_threshold = 0.60  # Lower threshold for long ranges
        elif date_range > 7:
            similarity_threshold = 0.63  # Slightly lower for medium ranges  
        else:
            similarity_threshold = settings.SIMILARITY_THRESHOLD  # Standard
            
        print(f"   Adaptive threshold: {similarity_threshold:.2f} for {date_range}-day range")
    else:
        similarity_threshold = settings.SIMILARITY_THRESHOLD
    
    # Pre-compute embeddings (existing code)
    all_articles = []
    article_to_bias = {}
    
    for bias, articles in bias_groups.items():
        for article in articles:
            all_articles.append(article)
            article_to_bias[len(all_articles) - 1] = bias
    
    texts = []
    for article in all_articles:
        text = f"{article.title}. {article.description or ''}".strip()
        if not text or text == ".":
            text = article
            '''