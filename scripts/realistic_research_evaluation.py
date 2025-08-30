# scripts/realistic_research_evaluation.py
import json
from pathlib import Path
import sys
import os

# Add config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.research_benchmarks import validate_metric

def find_latest_results():
    """Find the most recent evaluation results file"""
    data_dir = Path('data')
    if not data_dir.exists():
        return None
    
    result_files = list(data_dir.glob('research_evaluation_*.json'))
    if not result_files:
        return None
    
    return max(result_files, key=lambda x: x.stat().st_mtime)

def realistic_research_evaluation():
    """Research evaluation with realistic benchmarks and academic context"""
    
    print("REALISTIC RESEARCH EVALUATION")
    print("=" * 60)
    print("Using academic literature benchmarks and acknowledging limitations")
    print("")
    
    # Find latest results
    results_path = find_latest_results()
    if not results_path:
        print("❌ No evaluation results found. Run research evaluation first.")
        print("   Run: python scripts/run_research_evaluation.py --articles 300 --days 7")
        return
    
    print(f"📊 Using results from: {results_path.name}")
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    print("\n📈 REALISTIC PERFORMANCE ASSESSMENT")
    print("=" * 60)
    
    # RQ1: Technical Feasibility - Similarity Threshold
    rq1 = results.get('rq1_similarity_detection', {})
    threshold = rq1.get('optimal_threshold', 0)
    threshold_assessment = validate_metric('similarity', threshold)
    
    print(f"\n🔧 RQ1 - Similarity Threshold: {threshold:.3f}")
    print(f"   Assessment: {threshold_assessment['assessment']}")
    print(f"   Benchmarks: Excellent>{threshold_assessment['benchmarks']['excellent']:.2f}, "
          f"Good>{threshold_assessment['benchmarks']['good']:.2f}")
    
    # RQ2: Bias Detection
    rq2 = results.get('rq2_bias_detection', {})
    accuracy = rq2.get('bias_classification_results', {}).get('accuracy', 0)
    accuracy_assessment = validate_metric('bias_detection', accuracy)
    
    print(f"\n🎯 RQ2 - Bias Detection Accuracy: {accuracy:.3f}")
    print(f"   Assessment: {accuracy_assessment['assessment']}")
    
    if accuracy_assessment['concerns']:
        print("   ⚠️  CONCERNS:")
        for concern in accuracy_assessment['concerns']:
            print(f"     • {concern}")
    
    # RQ3: System Performance
    rq3 = results.get('rq3_system_performance', {})
    quality = rq3.get('quality_metrics', {}).get('composite_quality_score', 0)
    match_rate = rq3.get('performance_metrics', {}).get('match_rate', 0)
    
    quality_assessment = validate_metric('similarity', quality)
    match_assessment = validate_metric('match_success', match_rate)
    
    print(f"\n🚀 RQ3 - System Performance:")
    print(f"   Quality Score: {quality:.3f} - {quality_assessment['assessment']}")
    print(f"   Match Success Rate: {match_rate:.3f} - {match_assessment['assessment']}")
    
    # RQ4: User Impact
    rq4 = results.get('rq4_user_impact_simulation', {})
    diversity_improvement = rq4.get('overall_impact', {}).get('avg_diversity_improvement', 0)
    diversity_assessment = validate_metric('diversity_improvement', diversity_improvement)
    
    print(f"\n🌍 RQ4 - User Impact:")
    print(f"   Diversity Improvement: {diversity_improvement:.3f}")
    print(f"   Assessment: {diversity_assessment['assessment']}")
    
    # Academic Context
    print(f"\n📚 ACADEMIC CONTEXT")
    print("=" * 60)
    print("Bias Detection: " + validate_metric('bias_detection', 0)['academic_context'])
    print("\nSimilarity: " + validate_metric('similarity', 0)['academic_context'])
    
    # Research Recommendations
    print(f"\n💡 RESEARCH RECOMMENDATIONS")
    print("=" * 60)
    print("1. Conduct human validation of perspective matches")
    print("2. Acknowledge automated evaluation limitations in paper")
    print("3. Use academic benchmarks for realistic performance claims")
    print("4. Focus on technical feasibility demonstration")
    print("5. Consider smaller, more credible claims about impact")
    
    # Overall assessment
    print(f"\n✅ OVERALL RESEARCH STATUS")
    print("=" * 60)
    if (accuracy <= 0.85 and quality >= 0.65 and match_rate >= 0.05):
        print("READY FOR RESEARCH PAPER - Results are credible and defensible")
        print("Focus on demonstrated technical capabilities and realistic impact")
    else:
        print("NEEDS METHODOLOGICAL IMPROVEMENT - Address evaluation issues first")
        print("Consider: Human validation, better metrics, realistic expectations")

if __name__ == "__main__":
    realistic_research_evaluation()