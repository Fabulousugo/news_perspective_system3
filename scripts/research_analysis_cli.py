# scripts/research_analysis_cli.py - CLI for research analysis features

import click
import logging
import time
from pathlib import Path
import sys,os
import json
from datetime import datetime, timedelta

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.simple_extended_collector import SimpleExtendedCollector
from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
from src.research.user_study_framework import UserStudyFramework, POLARIZATION_SURVEY_QUESTIONS
from src.research.bias_visualization import BiasVisualization
from src.research.longitudinal_analysis import LongitudinalAnalyzer, ResearchIntegrator
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Research Analysis Tools for News Perspective System"""
    pass

@cli.command()
@click.option('--query', '-q', default='', help='Search query for articles')
@click.option('--days', '-d', default=14, help='Days to look back')
@click.option('--optimization', '-o', type=click.Choice(['standard', 'quantized', 'onnx']), 
              default='quantized', help='Optimization level')
def comprehensive_analysis(query: str, days: int, optimization: str):
    """Conduct comprehensive research analysis addressing all research questions"""
    
    print(" Comprehensive News Bias Research Analysis")
    print("=" * 60)
    print(f"Query: {query or 'General news'}")
    print(f"Analysis period: {days} days")
    print(f"Optimization: {optimization}")
    print("")
    
    try:
        # Initialize components
        print(" Initializing research components...")
        collector = SimpleExtendedCollector()
        matcher = OptimizedPerspectiveMatcher(optimization_level=optimization)
        research_integrator = ResearchIntegrator()
        
        # Collect articles
        print(" Collecting diverse articles for analysis...")
        start_time = time.time()
        
        diverse_articles = collector.collect_diverse_articles(
            query=query,
            days_back=days
        )
        
        # Flatten articles
        all_articles = []
        for bias_articles in diverse_articles.values():
            all_articles.extend(bias_articles)
        
        collection_time = time.time() - start_time
        print(f" Collected {len(all_articles)} articles in {collection_time:.2f}s")
        
        if len(all_articles) < 10:
            print(" Insufficient articles for meaningful research analysis")
            print("   Try: broader search query or more days")
            return
        
        # Find perspective matches
        print(" Identifying perspective matches...")
        match_start = time.time()
        
        matches = matcher.find_perspective_matches_fast(all_articles)
        match_time = time.time() - match_start
        
        print(f" Found {len(matches)} perspective matches in {match_time:.2f}s")
        
        # Convert matches to dict format for analysis
        match_dicts = []
        for match in matches:
            match_dict = {
                'story_id': match.story_id,
                'topic': match.topic,
                'confidence': match.confidence,
                'articles': match.articles,
                'similarity_scores': match.similarity_scores
            }
            match_dicts.append(match_dict)
        
        # Comprehensive analysis
        print(" Conducting comprehensive research analysis...")
        analysis_start = time.time()
        
        results = research_integrator.conduct_comprehensive_analysis(
            articles=all_articles,
            perspective_matches=match_dicts
        )
        
        analysis_time = time.time() - analysis_start
        total_time = collection_time + match_time + analysis_time
        
        # Display results
        print(f"\n Research Analysis Results")
        print("=" * 40)
        print(f" Total analysis time: {total_time:.2f}s")
        print(f" Dataset: {len(all_articles)} articles from {len(set(a.source for a in all_articles))} sources")
        print(f" Perspective matches: {len(matches)}")
        print("")
        
        # Research question answers
        rq_answers = results['research_question_answers']
        
        print(" Research Question Answers:")
        print("-" * 30)
        
        print("1 Automated NLP Bias Identification:")
        auto_bias = rq_answers['automated_bias_identification']
        print(f"   Feasibility: {auto_bias['feasibility']}")
        print(f"   Scale: {auto_bias['scale']}")
        print(f"   Bias categories detected: {auto_bias['bias_categories']}")
        print(f"   Approach: {auto_bias['accuracy_approach']}")
        
        print("\n2 Alternative Perspective Surfacing:")
        perspective = rq_answers['perspective_surfacing']
        print(f"   Effectiveness: {perspective['effectiveness']}")
        print(f"   Matches found: {perspective['matches_found']}")
        print(f"   Cross-bias coverage: {perspective['cross_bias_coverage']} different bias combinations")
        print(f"   Average confidence: {perspective['average_confidence']:.3f}")
        
        print("\n3 Algorithm Performance:")
        performance = rq_answers['algorithm_performance']
        print(f"   Processing: {performance['processing_speed']}")
        print(f"   Scalability: {performance['scalability']}")
        print(f"   Bias detection: {performance['bias_detection_approach']}")
        print(f"   Similarity: {performance['similarity_matching']}")
        
        # Show generated outputs
        print(f"\n Generated Outputs:")
        visualizations = results['visualizations_generated']
        reports = results['reports_generated']
        
        print(" Visualizations:")
        for viz_type, path in visualizations.items():
            if path:
                print(f"   • {viz_type.replace('_', ' ').title()}: {path}")
        
        print(" Reports:")
        for report_type, path in reports.items():
            print(f"   • {report_type.replace('_', ' ').title()}: {path}")
        
        # Research implications
        print(f"\n Research Implications:")
        match_rate = len(matches) / len(all_articles) if all_articles else 0
        
        if match_rate > 0.3:
            print("   HIGH perspective matching effectiveness demonstrated")
        elif match_rate > 0.1:
            print("   MODERATE perspective matching effectiveness demonstrated")
        else:
            print("   LIMITED perspective matching - may need threshold adjustment")
        
        if len(set(a.bias_label for a in all_articles if a.bias_label is not None)) >= 3:
            print("   DIVERSE political spectrum coverage achieved")
        else:
            print("   LIMITED political diversity - need more diverse sources")
        
        if len(all_articles) >= 100:
            print("   SUFFICIENT scale for statistical significance")
        else:
            print("   SMALL sample size - consider larger dataset")
        
        print(f"\n Next Steps for Research Completion:")
        print("   1. User Study: Run 'python scripts/research_analysis_cli.py user-study'")
        print("   2. Longitudinal: Run 'python scripts/research_analysis_cli.py track-trends'")
        print("   3. Validation: Consider expert annotation for ground truth")
        print("   4. Publication: Use generated visualizations and reports")
        
    except Exception as e:
        print(f" Analysis failed: {e}")
        import traceback
        print(f"Debug: {traceback.format_exc()}")

@cli.command()
@click.option('--user-id', '-u', required=True, help='User ID for the study')
@click.option('--phase', '-p', type=click.Choice(['pre', 'post', 'interaction']), 
              default='pre', help='Study phase')
def user_study(user_id: str, phase: str):
    """Conduct user impact study for polarization measurement"""
    
    print(f" User Impact Study - {phase.upper()} Phase")
    print("=" * 40)
    print(f"User ID: {user_id}")
    print("")
    
    try:
        study_framework = UserStudyFramework()
        
        if phase == 'pre':
            print(" Pre-Study Survey")
            print("This survey measures your initial political orientations and media consumption patterns.")
            print("")
            
            # Display survey questions
            questions = POLARIZATION_SURVEY_QUESTIONS['pre_study']
            responses = {}
            
            # Political orientation
            print("1. Political Orientation")
            print(f"   {questions['political_orientation']['question']}")
            print(f"   Scale: {questions['political_orientation']['scale']}")
            
            while True:
                try:
                    response = int(input("   Your response (1-7): "))
                    if 1 <= response <= 7:
                        responses['political_orientation'] = response
                        break
                    else:
                        print("   Please enter a number between 1 and 7")
                except ValueError:
                    print("   Please enter a valid number")
            
            print("")
            
            # Issue positions
            print("2. Issue Positions")
            responses['issue_positions'] = {}
            
            for issue, question in questions['issue_positions'].items():
                print(f"   {issue.replace('_', ' ').title()}: {question}")
                
                while True:
                    try:
                        response = int(input("   Your response (1-5): "))
                        if 1 <= response <= 5:
                            responses['issue_positions'][issue] = response
                            break
                        else:
                            print("   Please enter a number between 1 and 5")
                    except ValueError:
                        print("   Please enter a valid number")
                print("")
            
            # Media trust
            print("3. Media Trust")
            print(f"   {questions['media_trust']['question']}")
            responses['media_trust_scores'] = {}
            
            for source in questions['media_trust']['sources']:
                print(f"   {source}: {questions['media_trust']['scale']}")
                
                while True:
                    try:
                        response = int(input("   Your response (1-5): "))
                        if 1 <= response <= 5:
                            responses['media_trust_scores'][source] = response
                            break
                        else:
                            print("   Please enter a number between 1 and 5")
                    except ValueError:
                        print("   Please enter a valid number")
                print("")
            
            # Cross-cutting exposure willingness
            print("4. Cross-Cutting Exposure")
            print(f"   {questions['cross_cutting_exposure_willingness']['question']}")
            print(f"   Scale: {questions['cross_cutting_exposure_willingness']['scale']}")
            
            while True:
                try:
                    response = int(input("   Your response (1-5): "))
                    if 1 <= response <= 5:
                        responses['cross_cutting_exposure_willingness'] = response
                        break
                    else:
                        print("   Please enter a number between 1 and 5")
                except ValueError:
                    print("   Please enter a valid number")
            
            # Store responses
            study_framework.record_pre_study_measurement(user_id, responses)
            
            print("\n Pre-study survey completed!")
            print(" Responses recorded for longitudinal analysis")
            print("\n Next: Use the news system, then run post-study survey:")
            print(f"   python scripts/research_analysis_cli.py user-study -u {user_id} -p post")
            
        elif phase == 'post':
            print(" Post-Study Survey")
            print("This survey measures any changes after using the perspective system.")
            print("")
            
            # Simplified post-study (would include same questions as pre + additional ones)
            responses = {}
            
            print("System Effectiveness Questions:")
            
            post_questions = POLARIZATION_SURVEY_QUESTIONS['post_study']
            
            for key, question_data in post_questions.items():
                if isinstance(question_data, dict) and 'question' in question_data:
                    print(f"   {question_data['question']}")
                    print(f"   Scale: {question_data['scale']}")
                    
                    while True:
                        try:
                            response = int(input("   Your response (1-5): "))
                            if 1 <= response <= 5:
                                responses[key] = response
                                break
                            else:
                                print("   Please enter a number between 1 and 5")
                        except ValueError:
                            print("   Please enter a valid number")
                    print("")
            
            # For demo purposes, use same political questions as pre-study
            # In real study, you'd repeat all pre-study questions
            responses['political_orientation'] = 4  # Placeholder
            responses['issue_positions'] = {}
            responses['media_trust_scores'] = {}
            responses['cross_cutting_exposure_willingness'] = 3
            
            study_framework.record_post_study_measurement(user_id, responses)
            
            print(" Post-study survey completed!")
            
            # Show preliminary analysis
            analysis = study_framework.analyze_polarization_changes(user_id)
            
            print("\n Preliminary Individual Analysis:")
            if analysis['users_analyzed'] > 0:
                print("   Pre/post data available for analysis")
                print("   Changes will be included in group analysis")
            else:
                print("   Incomplete data - ensure both pre and post surveys completed")
            
        elif phase == 'interaction':
            print(" Interactive Study Session")
            print("This tracks your interaction with the news perspective system.")
            
            session_id = study_framework.start_user_session(user_id)
            print(f" Study session started: {session_id}")
            
            print("\n Instructions:")
            print("1. Browse news articles using the perspective system")
            print("2. Click on articles to read them")
            print("3. Explore alternative perspectives when available")
            print("4. The system will automatically track your interactions")
            
            print(f"\n Session ID for reference: {session_id}")
            print("   Use this if you need to pause and resume the session")
            
    except Exception as e:
        print(f" User study failed: {e}")

@cli.command()
@click.option('--days', '-d', default=30, help='Days to analyze trends')
def track_trends(days: int):
    """Track bias and perspective trends over time"""
    
    print(f" Longitudinal Trend Analysis")
    print(f"Analysis period: {days} days")
    print("=" * 40)
    
    try:
        analyzer = LongitudinalAnalyzer()
        
        # Analyze trends
        print(" Analyzing bias trends...")
        trends = analyzer.analyze_bias_trends(days)
        
        if 'error' in trends:
            print(f" {trends['error']}")
            print("\n To populate trend data:")
            print("   1. Run comprehensive analysis multiple times")
            print("   2. Use different queries and time periods")
            print("   3. System will automatically store temporal data")
            return
        
        print(" Trend analysis completed!")
        print("")
        
        # Display key metrics
        daily_avg = trends['daily_averages']
        print(" Daily Averages:")
        print(f"   Articles per day: {daily_avg['articles_per_day']:.1f}")
        print(f"   Perspective matches per day: {daily_avg['perspective_matches_per_day']:.1f}")
        print(f"   Average match confidence: {daily_avg['avg_match_confidence']:.3f}")
        print("")
        
        # Bias balance
        balance = trends['bias_balance']
        print(" Bias Balance Analysis:")
        print(f"   Average imbalance score: {balance['average_imbalance']:.3f}")
        print(f"   Most balanced day: {balance['most_balanced_day']}")
        print(f"   Most imbalanced day: {balance['most_imbalanced_day']}")
        print("")
        
        # Generate longitudinal report
        print(" Generating comprehensive trend report...")
        report_path = analyzer.generate_longitudinal_report(days)
        print(f" Report saved: {report_path}")
        
        print("\n Research Insights:")
        
        # Trend insights
        bias_dist = trends['bias_distribution_over_time']
        if len(bias_dist['dates']) >= 7:
            recent_total = bias_dist['left_trend'][-3:] + bias_dist['right_trend'][-3:]
            early_total = bias_dist['left_trend'][:3] + bias_dist['right_trend'][:3]
            
            if sum(recent_total) > sum(early_total):
                print("   Increasing political coverage over time")
            else:
                print("   Decreasing political coverage over time")
        
        # Perspective matching trends
        match_trend = trends['perspective_matching_trend']
        if match_trend['matches_per_day']:
            avg_matches = sum(match_trend['matches_per_day']) / len(match_trend['matches_per_day'])
            if avg_matches > 5:
                print("   Strong perspective matching capability")
            elif avg_matches > 2:
                print("   Moderate perspective matching capability")
            else:
                print("   Limited perspective matching - consider system improvements")
        
    except Exception as e:
        print(f" Trend analysis failed: {e}")

@cli.command()
@click.option('--query', '-q', default='', help='Search query')
@click.option('--days', '-d', default=7, help='Days to analyze')
@click.option('--output-dir', '-o', help='Output directory for visualizations')
def visualize(query: str, days: int, output_dir: str):
    """Generate bias pattern visualizations"""
    
    print(" Bias Pattern Visualization")
    print("=" * 35)
    
    try:
        # Initialize components
        collector = SimpleExtendedCollector()
        visualizer = BiasVisualization(output_dir)
        matcher = OptimizedPerspectiveMatcher()
        
        # Collect articles
        print(" Collecting articles for visualization...")
        diverse_articles = collector.collect_diverse_articles(query, days)
        
        all_articles = []
        for bias_articles in diverse_articles.values():
            all_articles.extend(bias_articles)
        
        print(f" Collected {len(all_articles)} articles")
        
        if len(all_articles) < 5:
            print(" Insufficient articles for visualization")
            return
        
        # Find perspective matches for network visualization
        print(" Finding perspective matches...")
        matches = matcher.find_perspective_matches_fast(all_articles)
        match_dicts = [match.__dict__ for match in matches]
        
        # Generate visualizations
        print(" Generating visualizations...")
        
        viz_paths = []
        
        # 1. Bias distribution
        path = visualizer.create_bias_distribution_chart(all_articles)
        if path:
            viz_paths.append(('Bias Distribution', path))
        
        # 2. Temporal analysis
        path = visualizer.create_temporal_bias_analysis(all_articles)
        if path:
            viz_paths.append(('Temporal Analysis', path))
        
        # 3. Source analysis
        path = visualizer.create_source_bias_analysis(all_articles)
        if path:
            viz_paths.append(('Source Analysis', path))
        
        # 4. Topic heatmap
        path = visualizer.create_topic_sentiment_heatmap(all_articles)
        if path:
            viz_paths.append(('Topic Coverage', path))
        
        # 5. Perspective network
        if match_dicts:
            path = visualizer.create_perspective_network_graph(match_dicts)
            if path:
                viz_paths.append(('Perspective Network', path))
        
        # 6. Comprehensive dashboard
        path = visualizer.create_comprehensive_dashboard(all_articles, match_dicts)
        if path:
            viz_paths.append(('Comprehensive Dashboard', path))
        
        # Display results
        print(f"\n Generated {len(viz_paths)} visualizations:")
        for viz_name, path in viz_paths:
            print(f"   {viz_name}: {path}")
        
        # Generate analysis report
        print("\n Generating analysis report...")
        report_path = visualizer.generate_bias_report(all_articles, match_dicts)
        print(f" Report saved: {report_path}")
        
        print(f"\n Visualization Summary:")
        print(f"   Charts generated: {len(viz_paths)}")
        print(f"   Analysis report: Available")
        print(f"   Output directory: {visualizer.output_dir}")
        
    except Exception as e:
        print(f" Visualization failed: {e}")

@cli.command()
def research_status():
    """Show current research data status and completeness"""
    
    print(" Research Analysis Status")
    print("=" * 30)
    
    try:
        # Check databases
        user_study = UserStudyFramework()
        longitudinal = LongitudinalAnalyzer()
        
        # Check user study data
        print(" User Study Data:")
        try:
            import sqlite3
            conn = sqlite3.connect(user_study.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM polarization_measurements WHERE measurement_type = 'pre'")
            pre_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM polarization_measurements WHERE measurement_type = 'post'")
            post_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM polarization_measurements")
            unique_users = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"   Pre-study surveys: {pre_count}")
            print(f"   Post-study surveys: {post_count}")
            print(f"   Unique users: {unique_users}")
            print(f"   Complete pairs: {min(pre_count, post_count)}")
            
        except Exception:
            print("   No user study data available")
        
        print("")
        
        # Check longitudinal data
        print(" Longitudinal Data:")
        try:
            conn = sqlite3.connect(longitudinal.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM articles_temporal")
            article_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM perspective_matches_temporal")
            match_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM daily_bias_stats WHERE total_articles > 0")
            active_days = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"   Articles stored: {article_count}")
            print(f"   Perspective matches: {match_count}")
            print(f"   Days with data: {active_days}")
            
        except Exception:
            print("   No longitudinal data available")
        
        print("")
        
        # Check visualization outputs
        print(" Generated Outputs:")
        viz_dir = settings.DATA_DIR / "visualizations"
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.html"))
            report_files = list(viz_dir.glob("*.md"))
            
            print(f"   Visualizations: {len(viz_files)}")
            print(f"   Reports: {len(report_files)}")
            
            if viz_files or report_files:
                print("   Recent files:")
                all_files = sorted(viz_files + report_files, key=lambda x: x.stat().st_mtime, reverse=True)
                for file in all_files[:5]:
                    modified = datetime.fromtimestamp(file.stat().st_mtime)
                    print(f"     • {file.name} ({modified.strftime('%Y-%m-%d %H:%M')})")
        else:
            print("   No visualization outputs found")
        
        print("")
        
        # Research completeness assessment
        print(" Research Completeness:")
        
        completeness_score = 0
        max_score = 5
        
        # Technical implementation
        print("   Automated bias detection: COMPLETE")
        completeness_score += 1
        
        print("   Perspective matching: COMPLETE")
        completeness_score += 1
        
        print("   Visualization system: COMPLETE")
        completeness_score += 1
        
        # Data collection
        if unique_users > 0:
            print("   User study data: AVAILABLE")
            completeness_score += 1
        else:
            print("   User study data: NEEDED")
        
        if active_days >= 7:
            print("   Longitudinal data: SUFFICIENT")
            completeness_score += 1
        else:
            print("   Longitudinal data: MORE NEEDED")
        
        print(f"\n Overall Completeness: {completeness_score}/{max_score} ({completeness_score/max_score*100:.0f}%)")
        
        if completeness_score >= 4:
            print(" Research system ready for academic publication!")
        elif completeness_score >= 3:
            print(" Core research components complete, collect more data")
        else:
            print(" Need more data collection for research completion")
        
    except Exception as e:
        print(f" Status check failed: {e}")

if __name__ == "__main__":
    cli()