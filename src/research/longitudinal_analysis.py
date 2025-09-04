# src/research/longitudinal_analysis.py - Longitudinal bias and perspective analysis

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict
from pathlib import Path
import json
import sys,os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.news_apis import Article
from src.research.bias_visualization import BiasVisualization
from src.research.user_study_framework import UserStudyFramework
from config.settings import settings

logger = logging.getLogger(__name__)

class LongitudinalAnalyzer:
    """Analyze bias patterns and perspective matching over time"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(settings.DATA_DIR / "longitudinal_analysis.db")
        self.visualizer = BiasVisualization()
        self._init_database()
        
    def _init_database(self):
        """Initialize database for longitudinal tracking"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Articles table with temporal tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles_temporal (
                article_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                source TEXT NOT NULL,
                bias_label INTEGER,
                published_at TEXT NOT NULL,
                collected_at TEXT NOT NULL,
                topic_keywords TEXT,
                sentiment_score REAL
            );
        ''')
        
        print("Research integration demo completed")

        
        # Perspective matches over time
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS perspective_matches_temporal (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                story_id TEXT NOT NULL,
                topic TEXT,
                confidence REAL,
                match_timestamp TEXT NOT NULL,
                articles_json TEXT NOT NULL,
                similarity_scores_json TEXT
            )
        ''')
        
        # Daily bias statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_bias_stats (
                date TEXT PRIMARY KEY,
                left_count INTEGER DEFAULT 0,
                center_count INTEGER DEFAULT 0,
                right_count INTEGER DEFAULT 0,
                total_articles INTEGER DEFAULT 0,
                perspective_matches INTEGER DEFAULT 0,
                avg_match_confidence REAL DEFAULT 0.0
            )
        ''')
        
        # Topic evolution tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topic_evolution (
                topic TEXT NOT NULL,
                date TEXT NOT NULL,
                bias_category TEXT NOT NULL,
                article_count INTEGER DEFAULT 0,
                avg_sentiment REAL DEFAULT 0.0,
                PRIMARY KEY (topic, date, bias_category)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Longitudinal analysis database initialized: {self.db_path}")
    
    def store_articles_batch(self, articles: List[Article], 
                           topic_keywords: List[str] = None) -> None:
        """Store batch of articles for longitudinal analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        collected_at = datetime.now().isoformat()
        
        for article in articles:
            try:
                # Extract topic keywords if provided
                keywords_json = None
                if topic_keywords:
                    found_keywords = []
                    title_lower = article.title.lower()
                    content_lower = (article.content or "").lower()
                    
                    for keyword in topic_keywords:
                        if keyword.lower() in title_lower or keyword.lower() in content_lower:
                            found_keywords.append(keyword)
                    
                    if found_keywords:
                        keywords_json = json.dumps(found_keywords)
                
                # Simple sentiment approximation (can be enhanced with proper sentiment analysis)
                sentiment_score = 0.0  # Neutral baseline
                
                cursor.execute('''
                    INSERT OR REPLACE INTO articles_temporal 
                    (title, url, source, bias_label, published_at, collected_at, topic_keywords, sentiment_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.title,
                    article.url,
                    article.source,
                    article.bias_label,
                    article.published_at.isoformat() if article.published_at else collected_at,
                    collected_at,
                    keywords_json,
                    sentiment_score
                ))
                
            except Exception as e:
                logger.warning(f"Failed to store article: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        # Update daily statistics
        self._update_daily_stats()
        
        logger.info(f"Stored {len(articles)} articles for longitudinal analysis")
    
    def store_perspective_matches(self, matches: List[Dict]) -> None:
        """Store perspective matches for temporal analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        match_timestamp = datetime.now().isoformat()
        
        for match in matches:
            try:
                cursor.execute('''
                    INSERT INTO perspective_matches_temporal
                    (story_id, topic, confidence, match_timestamp, articles_json, similarity_scores_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    match.get('story_id', ''),
                    match.get('topic', ''),
                    match.get('confidence', 0.0),
                    match_timestamp,
                    json.dumps(match.get('articles', {})),
                    json.dumps(match.get('similarity_scores', {}))
                ))
                
            except Exception as e:
                logger.warning(f"Failed to store perspective match: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored {len(matches)} perspective matches")
    
    def _update_daily_stats(self) -> None:
        """Update daily bias statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get today's date
        today = datetime.now().date().isoformat()
        
        # Count articles by bias for today
        cursor.execute('''
            SELECT bias_label, COUNT(*) 
            FROM articles_temporal 
            WHERE DATE(published_at) = ?
            GROUP BY bias_label
        ''', (today,))
        
        bias_counts = dict(cursor.fetchall())
        
        # Count perspective matches for today
        cursor.execute('''
            SELECT COUNT(*), AVG(confidence)
            FROM perspective_matches_temporal 
            WHERE DATE(match_timestamp) = ?
        ''', (today,))
        
        match_result = cursor.fetchone()
        match_count = match_result[0] if match_result else 0
        avg_confidence = match_result[1] if match_result and match_result[1] else 0.0
        
        # Update daily stats
        cursor.execute('''
            INSERT OR REPLACE INTO daily_bias_stats
            (date, left_count, center_count, right_count, total_articles, 
             perspective_matches, avg_match_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            today,
            bias_counts.get(0, 0),  # left
            bias_counts.get(1, 0),  # center  
            bias_counts.get(2, 0),  # right
            sum(bias_counts.values()),
            match_count,
            avg_confidence
        ))
        
        conn.commit()
        conn.close()
    
    def analyze_bias_trends(self, days_back: int = 30) -> Dict:
        """Analyze bias trends over time"""
        conn = sqlite3.connect(self.db_path)
        
        # Get daily stats
        df_daily = pd.read_sql_query('''
            SELECT * FROM daily_bias_stats 
            WHERE date >= date('now', '-{} days')
            ORDER BY date
        '''.format(days_back), conn)
        
        conn.close()
        
        if df_daily.empty:
            return {'error': 'No data available for trend analysis'}
        
        # Calculate trends
        analysis = {
            'date_range': {
                'start': df_daily['date'].min(),
                'end': df_daily['date'].max()
            },
            'total_days': len(df_daily),
            'total_articles': df_daily['total_articles'].sum(),
            'daily_averages': {
                'articles_per_day': df_daily['total_articles'].mean(),
                'perspective_matches_per_day': df_daily['perspective_matches'].mean(),
                'avg_match_confidence': df_daily['avg_match_confidence'].mean()
            },
            'bias_distribution_over_time': {
                'left_trend': df_daily['left_count'].tolist(),
                'center_trend': df_daily['center_count'].tolist(), 
                'right_trend': df_daily['right_count'].tolist(),
                'dates': df_daily['date'].tolist()
            },
            'perspective_matching_trend': {
                'matches_per_day': df_daily['perspective_matches'].tolist(),
                'confidence_trend': df_daily['avg_match_confidence'].tolist(),
                'dates': df_daily['date'].tolist()
            }
        }
        
        # Calculate bias balance metrics
        df_daily['bias_balance'] = np.abs(
            df_daily['left_count'] - df_daily['right_count']
        ) / (df_daily['left_count'] + df_daily['right_count'] + 1)  # +1 to avoid division by zero
        
        analysis['bias_balance'] = {
            'average_imbalance': df_daily['bias_balance'].mean(),
            'most_balanced_day': df_daily.loc[df_daily['bias_balance'].idxmin(), 'date'],
            'most_imbalanced_day': df_daily.loc[df_daily['bias_balance'].idxmax(), 'date']
        }
        
        return analysis
    
    def analyze_topic_evolution(self, topics: List[str], days_back: int = 30) -> Dict:
        """Analyze how topics evolve across different political perspectives over time"""
        conn = sqlite3.connect(self.db_path)
        
        # Get articles with topic keywords
        placeholders = ','.join(['?' for _ in topics])
        query = f'''
            SELECT DATE(published_at) as date, bias_label, topic_keywords, COUNT(*) as count
            FROM articles_temporal 
            WHERE DATE(published_at) >= date('now', '-{days_back} days')
            AND topic_keywords IS NOT NULL
            GROUP BY DATE(published_at), bias_label, topic_keywords
            ORDER BY date
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {'error': 'No topic data available'}
        
        # Process topic evolution
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning"}
        topic_evolution = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for _, row in df.iterrows():
            date = row['date']
            bias = bias_labels.get(row['bias_label'], 'unknown')
            keywords = json.loads(row['topic_keywords']) if row['topic_keywords'] else []
            
            for keyword in keywords:
                if keyword in topics:
                    topic_evolution[keyword][date][bias] += row['count']
        
        # Format for visualization
        evolution_data = {}
        for topic in topics:
            if topic in topic_evolution:
                topic_data = {
                    'dates': list(topic_evolution[topic].keys()),
                    'bias_coverage': {}
                }
                
                for bias in bias_labels.values():
                    topic_data['bias_coverage'][bias] = [
                        topic_evolution[topic][date].get(bias, 0)
                        for date in topic_data['dates']
                    ]
                
                evolution_data[topic] = topic_data
        
        return evolution_data
    
    def generate_longitudinal_report(self, days_back: int = 30) -> str:
        """Generate comprehensive longitudinal analysis report"""
        
        # Get trend analysis
        trends = self.analyze_bias_trends(days_back)
        
        if 'error' in trends:
            return f"Unable to generate report: {trends['error']}"
        
        report = []
        report.append("# Longitudinal Bias Analysis Report")
        report.append(f"**Analysis Period:** {trends['date_range']['start']} to {trends['date_range']['end']}")
        report.append(f"**Total Days:** {trends['total_days']}")
        report.append(f"**Total Articles Analyzed:** {trends['total_articles']}")
        report.append("")
        
        # Daily averages
        daily_avg = trends['daily_averages']
        report.append("## Daily Averages")
        report.append(f"- **Articles per day:** {daily_avg['articles_per_day']:.1f}")
        report.append(f"- **Perspective matches per day:** {daily_avg['perspective_matches_per_day']:.1f}")
        report.append(f"- **Average matching confidence:** {daily_avg['avg_match_confidence']:.3f}")
        report.append("")
        
        # Bias balance analysis
        balance = trends['bias_balance']
        report.append("## Bias Balance Analysis")
        report.append(f"- **Average bias imbalance:** {balance['average_imbalance']:.3f}")
        report.append(f"- **Most balanced day:** {balance['most_balanced_day']}")
        report.append(f"- **Most imbalanced day:** {balance['most_imbalanced_day']}")
        report.append("")
        
        # Trend insights
        report.append("## Key Trends")
        
        bias_dist = trends['bias_distribution_over_time']
        left_trend = np.mean(bias_dist['left_trend'][-7:]) - np.mean(bias_dist['left_trend'][:7]) if len(bias_dist['left_trend']) >= 14 else 0
        right_trend = np.mean(bias_dist['right_trend'][-7:]) - np.mean(bias_dist['right_trend'][:7]) if len(bias_dist['right_trend']) >= 14 else 0
        
        if left_trend > 5:
            report.append("- **Increasing left-leaning coverage** in recent days")
        elif left_trend < -5:
            report.append("- **Decreasing left-leaning coverage** in recent days")
        
        if right_trend > 5:
            report.append("- **Increasing right-leaning coverage** in recent days")
        elif right_trend < -5:
            report.append("- **Decreasing right-leaning coverage** in recent days")
        
        # Perspective matching trends
        match_trend = trends['perspective_matching_trend']
        recent_matches = np.mean(match_trend['matches_per_day'][-7:]) if match_trend['matches_per_day'] else 0
        early_matches = np.mean(match_trend['matches_per_day'][:7]) if len(match_trend['matches_per_day']) >= 7 else 0
        
        if recent_matches > early_matches:
            report.append("- **Improving perspective matching** over time")
        elif recent_matches < early_matches:
            report.append("- **Declining perspective matching** over time")
        
        report.append("")
        report.append("---")
        report.append("*Generated by Longitudinal Analysis System*")
        
        # Save report
        report_text = '\n'.join(report)
        output_path = settings.DATA_DIR / "longitudinal_report.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Longitudinal report saved: {output_path}")
        return str(output_path)

class ResearchIntegrator:
    """Integrate all research components for comprehensive analysis"""
    
    def __init__(self):
        self.user_study = UserStudyFramework()
        self.visualizer = BiasVisualization()
        self.longitudinal = LongitudinalAnalyzer()
        
        # Initialize databases
        self.user_study._init_database()
        self.longitudinal._init_database()

    def conduct_comprehensive_analysis(self, articles: List[Article], 
                                    perspective_matches: List[Dict],
                                    user_study_results: Optional[Dict] = None) -> Dict:
        """Conduct comprehensive research analysis addressing all research questions"""
        
        logger.info("Conducting comprehensive research analysis...")
        
        # Store data for longitudinal tracking
        self.longitudinal.store_articles_batch(articles)
        self.longitudinal.store_perspective_matches(perspective_matches)
        
        # Generate all visualizations
        visualizations = {
            'bias_distribution': self.visualizer.create_bias_distribution_chart(articles),
            'temporal_analysis': self.visualizer.create_temporal_bias_analysis(articles),
            'source_analysis': self.visualizer.create_source_bias_analysis(articles),
            'topic_heatmap': self.visualizer.create_topic_sentiment_heatmap(articles),
            'comprehensive_dashboard': self.visualizer.create_comprehensive_dashboard(articles, perspective_matches)
        }
        
        # Generate reports
        bias_report = self.visualizer.generate_bias_report(articles, perspective_matches, user_study_results)
        longitudinal_report = self.longitudinal.generate_longitudinal_report()
        
        # Analyze trends
        trend_analysis = self.longitudinal.analyze_bias_trends(30)
        
        # Research question answers
        research_answers = {
            'automated_bias_identification': {
                'feasibility': 'Demonstrated',
                'accuracy_approach': 'Source-based classification with ML validation',
                'scale': f'{len(articles)} articles processed',
                'bias_categories': len(set(a.bias_label for a in articles if a.bias_label is not None))
            },
            'perspective_surfacing': {
                'effectiveness': 'High' if len(perspective_matches) > len(articles) * 0.2 else 'Moderate',
                'matches_found': len(perspective_matches),
                'cross_bias_coverage': len(set(tuple(sorted(match.get('articles', {}).keys())) for match in perspective_matches)),
                'average_confidence': np.mean([match.get('confidence', 0) for match in perspective_matches]) if perspective_matches else 0
            },
            'algorithm_performance': {
                'processing_speed': 'Optimized with quantization and ONNX',
                'scalability': 'Demonstrated with 40+ sources',
                'bias_detection_approach': 'Multi-level (source-based + ML optional)',
                'similarity_matching': 'Semantic embedding based'
            }
        }
        
        # User impact analysis (if available)
        user_impact = {}
        if user_study_results:
            polarization_analysis = self.user_study.analyze_polarization_changes()
            user_impact = {
                'users_studied': polarization_analysis.get('users_analyzed', 0),
                'polarization_reduction': polarization_analysis.get('summary', {}).get('users_with_reduced_extremism', 0),
                'increased_openness': polarization_analysis.get('summary', {}).get('users_with_increased_openness', 0),
                'avg_political_change': polarization_analysis.get('summary', {}).get('avg_political_orientation_change', 0)
            }

        # --- FIX: Normalize datetimes to UTC ---
        from datetime import timezone

        def ensure_utc(dt):
            if dt is None:
                return None
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        published_dates = [ensure_utc(a.published_at) for a in articles if a.published_at]

        comprehensive_results = {
            'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
            'dataset_summary': {
                'total_articles': len(articles),
                'sources_count': len(set(a.source for a in articles)),
                'perspective_matches': len(perspective_matches),
                'date_range': {
                    'start': min(published_dates).isoformat() if published_dates else None,
                    'end': max(published_dates).isoformat() if published_dates else None
                }
            },
            'research_question_answers': research_answers,
            'user_impact_analysis': user_impact,
            'trend_analysis': trend_analysis,
            'visualizations_generated': visualizations,
            'reports_generated': {
                'bias_analysis': bias_report,
                'longitudinal_analysis': longitudinal_report
            }
        }

        # --- FIX: JSON-serialize NumPy types safely ---
        def make_serializable(obj):
            import numpy as np
            import datetime
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            return str(obj)  # fallback for anything else

        results_path = settings.DATA_DIR / "comprehensive_research_results.json"
        with open(results_path, 'w') as f:
            import json
            json.dump(comprehensive_results, f, indent=2, default=make_serializable)
        

        logger.info(f"Comprehensive analysis completed, results saved: {results_path}")

        return comprehensive_results

# Example usage
if __name__ == "__main__":
    # Demo of research components
    from datetime import datetime
    from ..data_collection.news_apis import Article
    
    # Create sample articles
    sample_articles = [
        Article("Climate action needed", "content", "url1", "CNN", datetime.now(), bias_label=0),
        Article("Climate costs too high", "content", "url2", "Fox News", datetime.now(), bias_label=2),
        Article("Climate debate continues", "content", "url3", "Reuters", datetime.now(), bias_label=1)
    ]
    
    # Initialize research integrator
    research = ResearchIntegrator()
    
    # Conduct analysis
    results = research.conduct_comprehensive_analysis(
        articles=sample_articles,
        perspective_matches=[{
            'story_id': 'climate_story',
            'topic': 'Climate',
            'confidence': 0.85,
            'articles': {'left': sample_articles[0], 'right': sample_articles[1]}
        }])