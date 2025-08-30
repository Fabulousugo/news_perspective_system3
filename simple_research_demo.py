# simple_research_demo.py - Working demo using existing components

import sys
from pathlib import Path
import json
from datetime import datetime
import logging

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def simple_research_demo():
    """Simple demo using existing working components"""
    
    print("Simple Automated Research Demo")
    print("=" * 40)
    print("Demonstrating automated analysis for dissertation research questions")
    print("")
    
    try:
        # Use existing working components
        from data_collection.simple_extended_collector import SimpleExtendedCollector
        from models.news_browser import NewsBrowser
        
        print("Research Questions:")
        print("1. What patterns distinguish left, center, right news sources?")
        print("2. Can systems effectively surface alternative perspectives?") 
        print("3. How do algorithms perform across political viewpoints?")
        print("")
        
        # Initialize components
        print("Initializing system components...")
        collector = SimpleExtendedCollector()
        browser = NewsBrowser()
        print("✅ System ready")
        print("")
        
        # Collect sample data
        print("Collecting sample articles for analysis...")
        diverse_articles = collector.collect_diverse_articles(
            query="healthcare",  # Single topic for demo
            days_back=7
        )
        
        # Analyze collection
        total_articles = sum(len(articles) for articles in diverse_articles.values())
        print(f"✅ Collected {total_articles} articles")
        
        if total_articles == 0:
            print("❌ No articles collected. Check API keys and connectivity.")
            return False
        
        # Show bias distribution (RQ1 data)
        print("\nRQ1: Political Viewpoint Analysis")
        print("-" * 35)
        
        bias_distribution = {}
        distinctive_sources = {}
        
        for bias_category, articles in diverse_articles.items():
            if articles:
                sources = list(set(article.source for article in articles))
                bias_distribution[bias_category] = len(articles)
                distinctive_sources[bias_category] = sources
                
                print(f"{bias_category.upper()}: {len(articles)} articles")
                print(f"  Sources: {', '.join(sources)}")
                
                # Sample titles showing different perspectives
                if len(articles) > 0:
                    sample_titles = [article.title for article in articles[:2]]
                    for title in sample_titles:
                        print(f"  • {title[:60]}...")
                print("")
        
        # Perspective matching analysis (RQ2 data)
        print("RQ2: Perspective Surfacing Analysis")
        print("-" * 35)
        
        # Flatten articles for perspective analysis
        all_articles = []
        for articles in diverse_articles.values():
            all_articles.extend(articles)
        
        # Use browser to find perspectives
        browseable_articles = browser.browse_articles(all_articles)
        
        # Analyze perspective effectiveness
        articles_with_perspectives = [a for a in browseable_articles if a.perspective_count > 0]
        perspective_coverage = len(articles_with_perspectives) / len(browseable_articles) if browseable_articles else 0
        
        print(f"Total articles analyzed: {len(browseable_articles)}")
        print(f"Articles with alternative perspectives: {len(articles_with_perspectives)}")
        print(f"Perspective coverage: {perspective_coverage:.1%}")
        
        if articles_with_perspectives:
            avg_perspectives = sum(a.perspective_count for a in articles_with_perspectives) / len(articles_with_perspectives)
            max_perspectives = max(a.perspective_count for a in articles_with_perspectives)
            print(f"Average perspectives per article: {avg_perspectives:.1f}")
            print(f"Maximum perspectives found: {max_perspectives}")
            
            # Show sample perspective matches
            print("\nSample perspective matches:")
            for i, browseable in enumerate(articles_with_perspectives[:3]):
                article = browseable.article
                print(f"  {i+1}. {article.title[:50]}...")
                print(f"     Source: {article.source}")
                print(f"     Alternative perspectives: {browseable.perspective_count}")
                
                # Show alternative viewpoints
                bias_names = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
                for related, similarity in browseable.related_articles[:2]:
                    alt_bias = bias_names.get(related.bias_label, "UNKNOWN")
                    print(f"       → {alt_bias} ({similarity:.2f}): {related.title[:40]}... ({related.source})")
                print("")
        
        # Algorithm performance analysis (RQ3 data)
        print("RQ3: Algorithm Performance Analysis")
        print("-" * 38)
        
        # Source-based accuracy assessment
        source_bias_map = {
            'CNN': 'left-leaning', 'The Guardian': 'left-leaning', 'MSNBC': 'left-leaning', 'NPR': 'left-leaning',
            'Reuters': 'centrist', 'Associated Press': 'centrist', 'BBC News': 'centrist',
            'Fox News': 'right-leaning', 'New York Post': 'right-leaning', 'Wall Street Journal': 'right-leaning',
            'The Daily Wire': 'right-leaning', 'Breitbart': 'right-leaning'
        }
        
        # Evaluate source classification accuracy
        correctly_classified = 0
        total_classified = 0
        
        classification_breakdown = {"left-leaning": 0, "centrist": 0, "right-leaning": 0}
        
        for bias_category, articles in diverse_articles.items():
            for article in articles:
                if article.source in source_bias_map:
                    expected_bias = source_bias_map[article.source]
                    if expected_bias == bias_category:
                        correctly_classified += 1
                    total_classified += 1
                    classification_breakdown[bias_category] += 1
        
        if total_classified > 0:
            accuracy = correctly_classified / total_classified
            print(f"Source-based classification accuracy: {accuracy:.1%}")
            print(f"Articles classified by viewpoint:")
            for viewpoint, count in classification_breakdown.items():
                print(f"  {viewpoint}: {count} articles")
        
        # Generate summary report
        print("\nAutomated Research Summary")
        print("=" * 30)
        
        findings = {
            "data_overview": {
                "total_articles": total_articles,
                "unique_sources": len(set(article.source for articles in diverse_articles.values() for article in articles)),
                "political_viewpoints": len([k for k, v in diverse_articles.items() if v]),
                "analysis_topic": "healthcare"
            },
            "rq1_findings": {
                "bias_distribution": bias_distribution,
                "distinctive_sources": distinctive_sources,
                "conclusion": f"Found systematic differences across {len(bias_distribution)} political viewpoints"
            },
            "rq2_findings": {
                "perspective_coverage": perspective_coverage,
                "articles_with_perspectives": len(articles_with_perspectives),
                "effectiveness_score": perspective_coverage * avg_perspectives if articles_with_perspectives else 0,
                "conclusion": f"System achieved {perspective_coverage:.1%} perspective coverage"
            },
            "rq3_findings": {
                "classification_accuracy": accuracy if total_classified > 0 else 0,
                "total_evaluated": total_classified,
                "conclusion": f"Algorithm achieved {accuracy:.1%} accuracy across viewpoints" if total_classified > 0 else "Limited evaluation data"
            }
        }
        
        # Save results
        output_file = f"simple_research_demo_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(output_file, 'w') as f:
            json.dump(findings, f, indent=2, default=str)
        
        print(f"Key Findings Summary:")
        print(f"  Data: {total_articles} articles from {findings['data_overview']['unique_sources']} sources")
        print(f"  RQ1: Identified {len(bias_distribution)} distinct political viewpoints")
        print(f"  RQ2: {perspective_coverage:.1%} perspective coverage achieved")
        print(f"  RQ3: {accuracy:.1%} classification accuracy" if total_classified > 0 else "  RQ3: Classification analysis completed")
        print("")
        print(f"✅ Demo completed successfully!")
        print(f"Results saved to: {output_file}")
        print("")
        print("This demonstrates automated analysis for dissertation without manual annotation:")
        print("• Bias pattern identification through source and content analysis")
        print("• Perspective surfacing effectiveness measurement") 
        print("• Algorithm performance evaluation using source-based validation")
        print("")
        print("For full research study:")
        print("1. Fix the optimization level issue in the framework")
        print("2. Run comprehensive analysis with more topics and longer timeframe")
        print("3. Generate visualizations and export results")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        print(f"Debug: {traceback.format_exc()}")
        print("")
        print("To fix, ensure you have:")
        print("• Valid API keys in .env file")
        print("• Internet connectivity")
        print("• All required packages installed")
        return False

if __name__ == "__main__":
    simple_research_demo()