# src/research/bias_visualization.py - Bias pattern visualization and temporal analysis

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict, Counter
from pathlib import Path

from ..data_collection.news_apis import Article
from config.settings import settings

logger = logging.getLogger(__name__)

class BiasVisualization:
    """Comprehensive bias pattern visualization and analysis"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or settings.DATA_DIR / "visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.bias_colors = {
            'left-leaning': '#1f77b4',    # Blue
            'centrist': '#2ca02c',        # Green  
            'right-leaning': '#d62728',   # Red
            'libertarian': '#ff7f0e',     # Orange
            'international': '#9467bd'    # Purple
        }
        
        logger.info(f"Bias visualization initialized, output dir: {self.output_dir}")
    
    def create_bias_distribution_chart(self, articles: List[Article], 
                                     title: str = "Political Bias Distribution") -> str:
        """Create bias distribution visualization"""
        
        # Count articles by bias
        bias_counts = defaultdict(int)
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning", 3: "libertarian"}
        
        for article in articles:
            if article.bias_label is not None:
                bias_name = bias_labels.get(article.bias_label, "unknown")
                bias_counts[bias_name] += 1
        
        # Create DataFrame
        df = pd.DataFrame(list(bias_counts.items()), columns=['Bias', 'Count'])
        
        # Create interactive plotly chart
        fig = px.pie(df, values='Count', names='Bias', 
                    title=title,
                    color_discrete_map=self.bias_colors)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            font_size=14,
            title_font_size=18,
            showlegend=True,
            height=500
        )
        
        # Save
        output_path = self.output_dir / "bias_distribution.html"
        fig.write_html(str(output_path))
        
        logger.info(f"Bias distribution chart saved: {output_path}")
        return str(output_path)
    
    def create_temporal_bias_analysis(self, articles: List[Article], 
                                    time_window: str = "daily") -> str:
        """Create temporal bias pattern analysis"""
        
        # Prepare data
        data = []
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning", 3: "libertarian"}
        
        for article in articles:
            if article.bias_label is not None and article.published_at:
                data.append({
                    'date': article.published_at.date(),
                    'bias': bias_labels.get(article.bias_label, "unknown"),
                    'source': article.source,
                    'title': article.title
                })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            logger.warning("No temporal data available for visualization")
            return ""
        
        # Group by time window
        if time_window == "daily":
            df['time_period'] = df['date']
        elif time_window == "weekly":
            df['time_period'] = df['date'].apply(lambda x: x.strftime('%Y-W%U'))
        
        # Count articles by bias and time
        temporal_counts = df.groupby(['time_period', 'bias']).size().reset_index(name='count')
        
        # Create stacked area chart
        fig = px.area(temporal_counts, x='time_period', y='count', color='bias',
                     title=f"Bias Distribution Over Time ({time_window.title()})",
                     color_discrete_map=self.bias_colors)
        
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Number of Articles",
            font_size=12,
            title_font_size=16,
            height=500
        )
        
        # Save
        output_path = self.output_dir / f"temporal_bias_{time_window}.html"
        fig.write_html(str(output_path))
        
        logger.info(f"Temporal bias analysis saved: {output_path}")
        return str(output_path)
    
    def create_perspective_network_graph(self, perspective_matches: List[Dict]) -> str:
        """Create network graph showing perspective connections"""
        
        # Prepare network data
        nodes = []
        edges = []
        node_ids = set()
        
        for match in perspective_matches:
            articles = match.get('articles', {})
            
            # Add nodes for each article
            for bias, article_data in articles.items():
                if isinstance(article_data, dict):
                    article = article_data
                else:
                    article = article_data.__dict__ if hasattr(article_data, '__dict__') else {}
                
                node_id = f"{article.get('source', 'Unknown')}_{len(node_ids)}"
                
                if node_id not in node_ids:
                    nodes.append({
                        'id': node_id,
                        'label': article.get('source', 'Unknown'),
                        'bias': bias,
                        'title': article.get('title', '')[:50] + "..."
                    })
                    node_ids.add(node_id)
            
            # Add edges between perspectives
            bias_keys = list(articles.keys())
            for i, bias1 in enumerate(bias_keys):
                for bias2 in bias_keys[i+1:]:
                    edges.append({
                        'from': f"{articles[bias1].get('source', 'Unknown') if isinstance(articles[bias1], dict) else articles[bias1].source}_{i}",
                        'to': f"{articles[bias2].get('source', 'Unknown') if isinstance(articles[bias2], dict) else articles[bias2].source}_{i+1}",
                        'weight': match.get('confidence', 0.5)
                    })
        
        # Create network visualization using plotly
        # Simplified network as plotly doesn't have native network graphs
        if nodes and edges:
            # Create a matrix representation for heatmap
            sources = [node['label'] for node in nodes]
            bias_matrix = np.zeros((len(sources), len(sources)))
            
            # Fill matrix based on connections
            source_to_idx = {source: idx for idx, source in enumerate(sources)}
            
            for edge in edges:
                # Simplified: just mark connections
                try:
                    from_source = edge['from'].split('_')[0]
                    to_source = edge['to'].split('_')[0]
                    
                    if from_source in source_to_idx and to_source in source_to_idx:
                        from_idx = source_to_idx[from_source]
                        to_idx = source_to_idx[to_source]
                        bias_matrix[from_idx][to_idx] = edge['weight']
                        bias_matrix[to_idx][from_idx] = edge['weight']
                except:
                    continue
            
            # Create heatmap
            fig = px.imshow(bias_matrix, 
                           x=sources, y=sources,
                           title="Source Cross-Perspective Connections",
                           aspect="auto")
            
            fig.update_layout(
                title_font_size=16,
                height=600,
                width=800
            )
            
            output_path = self.output_dir / "perspective_network.html"
            fig.write_html(str(output_path))
            
            logger.info(f"Perspective network graph saved: {output_path}")
            return str(output_path)
        
        return ""
    
    def create_topic_sentiment_heatmap(self, articles: List[Article], 
                                     topic_keywords: List[str] = None) -> str:
        """Create topic-bias sentiment heatmap"""
        
        if not topic_keywords:
            topic_keywords = ['climate', 'election', 'economy', 'healthcare', 'immigration', 'education']
        
        # Analyze topics across bias categories
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning", 3: "libertarian"}
        
        topic_bias_matrix = defaultdict(lambda: defaultdict(int))
        
        for article in articles:
            if article.bias_label is not None:
                bias_name = bias_labels.get(article.bias_label, "unknown")
                title_lower = article.title.lower()
                content_lower = (article.content or "").lower()
                
                # Check for topic keywords
                for topic in topic_keywords:
                    if topic.lower() in title_lower or topic.lower() in content_lower:
                        topic_bias_matrix[topic][bias_name] += 1
        
        # Convert to matrix format
        topics = list(topic_bias_matrix.keys())
        biases = list(set(bias_labels.values()))
        
        if topics and biases:
            matrix_data = []
            for topic in topics:
                row = []
                for bias in biases:
                    count = topic_bias_matrix[topic][bias]
                    row.append(count)
                matrix_data.append(row)
            
            # Create heatmap
            fig = px.imshow(matrix_data,
                           x=biases,
                           y=topics,
                           title="Topic Coverage by Political Bias",
                           aspect="auto",
                           color_continuous_scale="Viridis")
            
            fig.update_layout(
                title_font_size=16,
                xaxis_title="Political Bias",
                yaxis_title="Topics",
                height=500
            )
            
            output_path = self.output_dir / "topic_sentiment_heatmap.html"
            fig.write_html(str(output_path))
            
            logger.info(f"Topic sentiment heatmap saved: {output_path}")
            return str(output_path)
        
        return ""
    
    def create_source_bias_analysis(self, articles: List[Article]) -> str:
        """Create source-level bias analysis"""
        
        # Group articles by source and analyze bias distribution
        source_data = defaultdict(lambda: defaultdict(int))
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning", 3: "libertarian"}
        
        for article in articles:
            if article.bias_label is not None:
                bias_name = bias_labels.get(article.bias_label, "unknown")
                source_data[article.source][bias_name] += 1
        
        # Create DataFrame for visualization
        data = []
        for source, bias_counts in source_data.items():
            total_articles = sum(bias_counts.values())
            for bias, count in bias_counts.items():
                percentage = (count / total_articles) * 100 if total_articles > 0 else 0
                data.append({
                    'source': source,
                    'bias': bias,
                    'count': count,
                    'percentage': percentage
                })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            # Create stacked bar chart
            fig = px.bar(df, x='source', y='percentage', color='bias',
                        title="Source Bias Distribution",
                        color_discrete_map=self.bias_colors)
            
            fig.update_layout(
                xaxis_title="News Source",
                yaxis_title="Percentage of Articles",
                title_font_size=16,
                xaxis={'categoryorder': 'total descending'},
                height=600
            )
            
            # Rotate x-axis labels for readability
            fig.update_xaxes(tickangle=45)
            
            output_path = self.output_dir / "source_bias_analysis.html"
            fig.write_html(str(output_path))
            
            logger.info(f"Source bias analysis saved: {output_path}")
            return str(output_path)
        
        return ""
    
    def create_comprehensive_dashboard(self, articles: List[Article], 
                                    perspective_matches: List[Dict] = None) -> str:
        """Create comprehensive bias analysis dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Bias Distribution', 'Timeline Analysis', 
                           'Source Analysis', 'Topic Coverage'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning", 3: "libertarian"}
        
        # 1. Bias distribution pie chart
        bias_counts = defaultdict(int)
        for article in articles:
            if article.bias_label is not None:
                bias_name = bias_labels.get(article.bias_label, "unknown")
                bias_counts[bias_name] += 1
        
        fig.add_trace(
            go.Pie(labels=list(bias_counts.keys()), 
                  values=list(bias_counts.values()),
                  name="Bias Distribution"),
            row=1, col=1
        )
        
        # 2. Timeline analysis
        if articles:
            dates = [article.published_at.date() for article in articles if article.published_at]
            if dates:
                date_counts = Counter(dates)
                fig.add_trace(
                    go.Scatter(x=list(date_counts.keys()), 
                             y=list(date_counts.values()),
                             mode='lines+markers',
                             name="Articles Over Time"),
                    row=1, col=2
                )
        
        # 3. Source analysis
        source_counts = Counter([article.source for article in articles])
        top_sources = dict(source_counts.most_common(10))
        
        fig.add_trace(
            go.Bar(x=list(top_sources.keys()), 
                  y=list(top_sources.values()),
                  name="Top Sources"),
            row=2, col=1
        )
        
        # 4. Basic topic coverage (simplified)
        topics = ['climate', 'election', 'economy', 'healthcare']
        topic_counts = []
        for topic in topics:
            count = sum(1 for article in articles 
                       if topic.lower() in article.title.lower())
            topic_counts.append(count)
        
        fig.add_trace(
            go.Bar(x=topics, y=topic_counts, name="Topic Coverage"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="News Bias Analysis Dashboard",
            title_font_size=20,
            showlegend=False,
            height=800
        )
        
        output_path = self.output_dir / "comprehensive_dashboard.html"
        fig.write_html(str(output_path))
        
        logger.info(f"Comprehensive dashboard saved: {output_path}")
        return str(output_path)
    
    def generate_bias_report(self, articles: List[Article], 
                           perspective_matches: List[Dict] = None,
                           user_study_results: Dict = None) -> str:
        """Generate comprehensive bias analysis report"""
        
        report = []
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning", 3: "libertarian"}
        
        # Basic statistics
        total_articles = len(articles)
        bias_counts = defaultdict(int)
        source_counts = defaultdict(int)
        
        for article in articles:
            if article.bias_label is not None:
                bias_name = bias_labels.get(article.bias_label, "unknown")
                bias_counts[bias_name] += 1
            source_counts[article.source] += 1
        
        report.append("# News Bias Analysis Report")
        report.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"**Total articles analyzed:** {total_articles}")
        report.append("")
        
        # Bias distribution
        report.append("## Political Bias Distribution")
        for bias, count in bias_counts.items():
            percentage = (count / total_articles) * 100 if total_articles > 0 else 0
            report.append(f"- **{bias.title()}:** {count} articles ({percentage:.1f}%)")
        report.append("")
        
        # Source analysis
        report.append("## Source Analysis")
        report.append(f"**Total sources:** {len(source_counts)}")
        report.append("**Top 10 sources by article count:**")
        
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for source, count in top_sources:
            percentage = (count / total_articles) * 100 if total_articles > 0 else 0
            report.append(f"- {source}: {count} articles ({percentage:.1f}%)")
        report.append("")
        
        # Perspective matching analysis
        if perspective_matches:
            report.append("## Perspective Matching Analysis")
            report.append(f"**Total perspective matches found:** {len(perspective_matches)}")
            
            # Calculate cross-bias coverage
            cross_bias_pairs = defaultdict(int)
            for match in perspective_matches:
                articles_in_match = match.get('articles', {})
                bias_types = list(articles_in_match.keys())
                if len(bias_types) >= 2:
                    bias_pair = '-'.join(sorted(bias_types))
                    cross_bias_pairs[bias_pair] += 1
            
            report.append("**Cross-bias perspective coverage:**")
            for pair, count in cross_bias_pairs.items():
                report.append(f"- {pair}: {count} matches")
            
            # Average confidence
            confidences = [match.get('confidence', 0) for match in perspective_matches]
            avg_confidence = np.mean(confidences) if confidences else 0
            report.append(f"**Average matching confidence:** {avg_confidence:.3f}")
            report.append("")
        
        # Temporal analysis
        if articles:
            dates = [article.published_at.date() for article in articles if article.published_at]
            if dates:
                date_range = max(dates) - min(dates)
                report.append("## Temporal Coverage")
                report.append(f"**Date range:** {min(dates)} to {max(dates)} ({date_range.days} days)")
                
                # Articles per day
                daily_counts = Counter(dates)
                avg_daily = len(articles) / len(daily_counts) if daily_counts else 0
                report.append(f"**Average articles per day:** {avg_daily:.1f}")
                report.append("")
        
        # User study results
        if user_study_results:
            report.append("## User Impact Analysis")
            
            users_analyzed = user_study_results.get('users_analyzed', 0)
            if users_analyzed > 0:
                report.append(f"**Users studied:** {users_analyzed}")
                
                # Polarization changes
                pol_change = user_study_results.get('summary', {}).get('avg_political_orientation_change', 0)
                cross_change = user_study_results.get('summary', {}).get('avg_cross_cutting_willingness_change', 0)
                
                report.append(f"**Average political orientation change:** {pol_change:.3f}")
                report.append(f"**Average cross-cutting exposure willingness change:** {cross_change:.3f}")
                
                # Interpretation
                if cross_change > 0:
                    report.append("*Users showed increased willingness to view opposing perspectives*")
                if abs(pol_change) < 0.1:
                    report.append("*Political orientations remained relatively stable*")
                
                report.append("")
        
        # Key findings
        report.append("## Key Findings")
        
        # Bias diversity
        bias_diversity = len(bias_counts)
        if bias_diversity >= 3:
            report.append("- **High bias diversity:** Articles represent multiple political perspectives")
        elif bias_diversity == 2:
            report.append("- **Moderate bias diversity:** Articles represent two political perspectives")
        else:
            report.append("- **Limited bias diversity:** Articles primarily from one political perspective")
        
        # Source diversity
        source_diversity = len(source_counts)
        if source_diversity >= 10:
            report.append("- **High source diversity:** Wide range of news outlets represented")
        elif source_diversity >= 5:
            report.append("- **Moderate source diversity:** Several different news outlets")
        else:
            report.append("- **Limited source diversity:** Few news sources represented")
        
        # Perspective matching effectiveness
        if perspective_matches:
            match_rate = len(perspective_matches) / total_articles if total_articles > 0 else 0
            if match_rate > 0.3:
                report.append("- **Effective perspective matching:** High rate of cross-perspective story identification")
            elif match_rate > 0.1:
                report.append("- **Moderate perspective matching:** Some cross-perspective stories identified")
            else:
                report.append("- **Limited perspective matching:** Few cross-perspective connections found")
        
        report.append("")
        report.append("---")
        report.append("*Report generated by News Perspective Analysis System*")
        
        # Save report
        report_text = '\n'.join(report)
        output_path = self.output_dir / "bias_analysis_report.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Bias analysis report saved: {output_path}")
        return str(output_path)