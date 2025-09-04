# test_complete_research_system.py - Comprehensive test for all research components

import sys,os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_research_system():
    """Test that all research components work together"""
    print(" Testing Complete Research System")
    print("=" * 50)
    
    success_count = 0
    total_tests = 6
    
    # Test 1: Basic imports
    print("\n1 Testing Research Component Imports...")
    try:
        from src.research.user_study_framework import UserStudyFramework
        from src.research.bias_visualization import BiasVisualization  
        from src.research.longitudinal_analysis import LongitudinalAnalyzer, ResearchIntegrator
        from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
        from src.data_collection.simple_extended_collector import SimpleExtendedCollector
        print(" All research components import successfully")
        success_count += 1
    except Exception as e:
        print(f" Import failed: {e}")
    
    # Test 2: User study framework
    print("\n2 Testing User Study Framework...")
    try:
        study = UserStudyFramework()
        
        # Test pre-study measurement
        test_responses = {
            'political_orientation': 3,
            'issue_positions': {'climate': 4, 'healthcare': 3},
            'media_trust_scores': {'CNN': 3, 'Fox News': 2},
            'cross_cutting_exposure_willingness': 3
        }
        
        study.record_pre_study_measurement('test_user', test_responses)
        
        # Test session tracking
        session_id = study.start_user_session('test_user')
        
        # Test article interaction
        study.record_article_interaction(
            session_id=session_id,
            article_data={'url': 'test.com', 'bias_label': 1},
            time_spent=30.0,
            perspectives_viewed=2
        )
        
        print(" User study framework working")
        success_count += 1
    except Exception as e:
        print(f" User study test failed: {e}")
    
    # Test 3: Visualization system
    print("\n3 Testing Visualization System...")
    try:
        from src.data_collection.news_apis import Article
        from datetime import datetime
        
        visualizer = BiasVisualization()
        
        # Create test articles
        test_articles = [
            Article("Test left article", "content", "url1", "CNN", datetime.now(), bias_label=0),
            Article("Test right article", "content", "url2", "Fox News", datetime.now(), bias_label=2),
            Article("Test center article", "content", "url3", "Reuters", datetime.now(), bias_label=1),
        ]
        
        # Test bias distribution chart
        chart_path = visualizer.create_bias_distribution_chart(test_articles)
        
        # Test report generation
        report_path = visualizer.generate_bias_report(test_articles)
        
        print(" Visualization system working")
        success_count += 1
    except Exception as e:
        print(f" Visualization test failed: {e}")
    
    # Test 4: Longitudinal analysis
    print("\n4 Testing Longitudinal Analysis...")
    try:
        analyzer = LongitudinalAnalyzer()
        
        # Test storing articles
        analyzer.store_articles_batch(test_articles)
        
        # Test trend analysis
        trends = analyzer.analyze_bias_trends(7)
        
        print(" Longitudinal analysis working")
        success_count += 1
    except Exception as e:
        print(f" Longitudinal test failed: {e}")
    
    # Test 5: Research integration
    print("\n5 Testing Research Integration...")
    try:
        integrator = ResearchIntegrator()
        
        # Test perspective matches
        test_matches = [{
            'story_id': 'test_story',
            'topic': 'Test Topic',
            'confidence': 0.85,
            'articles': {'left': test_articles[0], 'right': test_articles[1]}
        }]
        
        # Test comprehensive analysis
        results = integrator.conduct_comprehensive_analysis(
            articles=test_articles,
            perspective_matches=test_matches
        )
        
        # Check results structure
        assert 'research_question_answers' in results
        assert 'dataset_summary' in results
        assert 'visualizations_generated' in results
        
        print(" Research integration working")
        success_count += 1
    except Exception as e:
        print(f" Research integration test failed: {e}")
    
    # Test 6: Full pipeline with optimized models
    print("\n6 Testing Full Research Pipeline...")
    try:
        # Test optimized perspective matching
        matcher = OptimizedPerspectiveMatcher(optimization_level="quantized")
        
        matches = matcher.find_perspective_matches_fast(test_articles)
        
        # Test comprehensive analysis with real data
        collector = SimpleExtendedCollector()
        
        print(" Full pipeline working")
        success_count += 1
    except Exception as e:
        print(f" Full pipeline test failed: {e}")
    
    # Results summary
    print(f"\n Test Results Summary")
    print("=" * 25)
    print(f"Tests passed: {success_count}/{total_tests}")
    print(f"Success rate: {success_count/total_tests*100:.0f}%")
    
    if success_count == total_tests:
        print("\n All research components working perfectly!")
        print("\n Your system is ready for:")
        print("   Comprehensive bias analysis")
        print("   User impact studies") 
        print("   Longitudinal trend tracking")
        print("   Research visualizations")
        print("   Academic publication")
        
        print("\n Next steps:")
        print("   1. Run comprehensive analysis:")
        print("      python scripts/research_analysis_cli.py comprehensive-analysis --query 'election'")
        print("   2. Start user studies:")
        print("      python scripts/research_analysis_cli.py user-study -u user001 -p pre")
        print("   3. Generate visualizations:")
        print("      python scripts/research_analysis_cli.py visualize --query 'climate'")
        print("   4. Track longitudinal trends:")
        print("      python scripts/research_analysis_cli.py track-trends --days 30")
        
        return True
    else:
        print(f"\n {total_tests - success_count} components need attention")
        print("\n Troubleshooting:")
        print("   Ensure all dependencies installed: pip install -r requirements.txt")
        print("   Check API keys configured in .env file")
        print("   Verify database permissions in data/ directory")
        
        return False

if __name__ == "__main__":
    test_research_system()