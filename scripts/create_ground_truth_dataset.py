# scripts/create_ground_truth_dataset.py
"""
Creates methodologically sound ground truth for automated evaluation
using objective, verifiable criteria instead of human subjectivity.
"""
import json
from pathlib import Path
from datetime import datetime

def create_ground_truth_dataset():
    """Create objective ground truth using verifiable criteria"""
    
    print("Creating Methodologically Sound Ground Truth Dataset")
    print("=" * 60)
    print("Using objective criteria instead of human subjectivity")
    print("")
    
    # Objective ground truth based on verifiable criteria
    ground_truth = {
        "creation_date": datetime.now().isoformat(),
        "methodology": "objective_criteria_verifiable_metrics",
        "evaluation_principles": [
            "Use identical story pairs with known perspective differences",
            "Focus on verifiable technical metrics, not subjective quality",
            "Measure what can be objectively measured",
            "Avoid subjective human judgment entirely"
        ],
        "test_cases": [
            {
                "test_id": "tc_001",
                "description": "Same press release, different editorial framing",
                "left_article": {
                    "source": "verified_left_source",
                    "title_template": "Policy Announcement: {topic} - Progressive Benefits",
                    "content_template": "The administration announced {policy} today, highlighting benefits for {beneficiaries}."
                },
                "right_article": {
                    "source": "verified_right_source", 
                    "title_template": "Policy Analysis: {topic} - Economic Impacts",
                    "content_template": "The government revealed {policy} today, raising questions about {economic_concerns}."
                },
                "expected_outcome": {
                    "should_match": True,
                    "perspective_difference": True,
                    "core_facts_same": True
                }
            }
        ],
        "evaluation_metrics": {
            "technical_metrics": [
                "precision_at_k",
                "recall_at_k", 
                "f1_score",
                "processing_throughput",
                "match_consistency"
            ],
            "banned_metrics": [
                "subjective_quality_score",
                "human_like_assessment", 
                "unvalidated_impact_claims"
            ]
        },
        "validation_method": "automated_cross_verification",
        "success_criteria": [
            "F1 score > 0.6 on verified test cases",
            "False positive rate < 0.2",
            "Processing speed > 30 articles/second",
            "Perspective detection accuracy > 0.7 on source verification"
        ]
    }
    
    # Save ground truth
    output_path = Path('data/ground_truth_dataset.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"✅ Created objective ground truth: {output_path}")
    print("")
    print("🎯 EVALUATION PRINCIPLES:")
    print("• Measure technical capabilities, not subjective quality")
    print("• Use verifiable criteria, not human opinion")
    print("• Focus on what can be objectively measured")
    print("• Avoid all subjective human judgment")

if __name__ == "__main__":
    create_ground_truth_dataset()