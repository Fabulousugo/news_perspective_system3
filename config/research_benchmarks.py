# config/research_benchmarks.py
"""
Realistic performance benchmarks based on academic literature
for news perspective analysis systems.
"""

# Based on political bias detection literature (Baly et al., 2020; Horne et al., 2018)
BIAS_DETECTION_BENCHMARKS = {
    'excellent': 0.85,  # State-of-the-art performance
    'good': 0.75,       # Competitive research performance  
    'fair': 0.65,       # Adequate for proof-of-concept
    'poor': 0.55        # Below research standards
}

# Based on semantic similarity research (Reimers et al., 2019; Cer et al., 2018)
SIMILARITY_BENCHMARKS = {
    'excellent': 0.80,  # High-quality semantic matching
    'good': 0.70,       # Reliable matching
    'fair': 0.60,       # Basic functionality
    'poor': 0.50        # Limited effectiveness
}

# Based on information retrieval and perspective finding literature
MATCH_SUCCESS_BENCHMARKS = {
    'excellent': 0.15,  # 15% of articles find perspective matches
    'good': 0.10,       # 10% match rate
    'fair': 0.05,       # 5% match rate  
    'poor': 0.02        # 2% match rate
}

# Based on user study literature on diversity interventions
DIVERSITY_IMPROVEMENT_BENCHMARKS = {
    'excellent': 0.25,  # 25% average improvement
    'good': 0.15,       # 15% improvement
    'fair': 0.08,       # 8% improvement
    'poor': 0.03        # 3% improvement
}

def get_realistic_assessment(metric_type: str, value: float) -> str:
    """
    Get research-appropriate assessment based on academic literature benchmarks
    
    Args:
        metric_type: Type of metric ('bias_detection', 'similarity', 
                     'match_success', 'diversity_improvement')
        value: The metric value to assess
        
    Returns:
        Assessment string with research context
    """
    if metric_type == 'bias_detection':
        benchmarks = BIAS_DETECTION_BENCHMARKS
        metric_name = "Bias Detection Accuracy"
    elif metric_type == 'similarity':
        benchmarks = SIMILARITY_BENCHMARKS
        metric_name = "Similarity Performance"
    elif metric_type == 'match_success':
        benchmarks = MATCH_SUCCESS_BENCHMARKS
        metric_name = "Match Success Rate"
    elif metric_type == 'diversity_improvement':
        benchmarks = DIVERSITY_IMPROVEMENT_BENCHMARKS
        metric_name = "Diversity Improvement"
    else:
        return "Unknown metric type"
    
    if value >= benchmarks['excellent']:
        return f"EXCELLENT ({value:.3f}) - State-of-the-art {metric_name}"
    elif value >= benchmarks['good']:
        return f"GOOD ({value:.3f}) - Competitive research performance"
    elif value >= benchmarks['fair']:
        return f"FAIR ({value:.3f}) - Adequate proof-of-concept"
    else:
        return f"NEEDS IMPROVEMENT ({value:.3f}) - Below research standards"

def get_academic_context(metric_type: str) -> str:
    """Get academic literature context for each metric"""
    contexts = {
        'bias_detection': (
            "Based on political bias detection literature (Baly et al., 2020; "
            "Horne et al., 2018). State-of-the-art systems achieve 80-85% accuracy "
            "on challenging news datasets."
        ),
        'similarity': (
            "Based on semantic similarity research (Reimers et al., 2019). "
            "High-quality sentence transformers achieve 0.75-0.85 cosine similarity "
            "for related news stories."
        ),
        'match_success': (
            "Based on perspective finding literature. In real-world news ecosystems, "
            "10-15% of articles have clear cross-perspective matches due to topic "
            "coverage and timing factors."
        ),
        'diversity_improvement': (
            "Based on diversity intervention studies (Munson et al., 2013). "
            "Successful systems achieve 15-25% diversity improvement in "
            "controlled user studies."
        )
    }
    return contexts.get(metric_type, "No academic context available.")

def validate_metric(metric_type: str, value: float) -> dict:
    """
    Comprehensive metric validation with academic context
    
    Returns:
        Dictionary with assessment, benchmarks, and academic context
    """
    return {
        'value': value,
        'assessment': get_realistic_assessment(metric_type, value),
        'benchmarks': {
            'excellent': BIAS_DETECTION_BENCHMARKS['excellent'] if metric_type == 'bias_detection' else
                        SIMILARITY_BENCHMARKS['excellent'] if metric_type == 'similarity' else
                        MATCH_SUCCESS_BENCHMARKS['excellent'] if metric_type == 'match_success' else
                        DIVERSITY_IMPROVEMENT_BENCHMARKS['excellent'],
            'good': BIAS_DETECTION_BENCHMARKS['good'] if metric_type == 'bias_detection' else
                    SIMILARITY_BENCHMARKS['good'] if metric_type == 'similarity' else
                    MATCH_SUCCESS_BENCHMARKS['good'] if metric_type == 'match_success' else
                    DIVERSITY_IMPROVEMENT_BENCHMARKS['good'],
            'fair': BIAS_DETECTION_BENCHMARKS['fair'] if metric_type == 'bias_detection' else
                    SIMILARITY_BENCHMARKS['fair'] if metric_type == 'similarity' else
                    MATCH_SUCCESS_BENCHMARKS['fair'] if metric_type == 'match_success' else
                    DIVERSITY_IMPROVEMENT_BENCHMARKS['fair']
        },
        'academic_context': get_academic_context(metric_type),
        'concerns': [] if value <= 0.9 else [
            "Value exceeds realistic research expectations",
            "May indicate methodological issues or data leakage",
            "Recommend human validation for verification"
        ]
    }