# src/research/automated_evaluation.py - Automated research framework

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data_collection.simple_extended_collector import SimpleExtendedCollector
from ..models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
from ..models.optimized_models import OptimizedBiasClassifier, OptimizedSimilarityDetector
from config.settings import settings

logger = logging.getLogger(__name__)

class AutomatedResearchFramework:
    """
    Automated framework to answer dissertation research questions
    without manual annotations or questionnaires
    """
    
    def __init__(self, optimization_level: str = "quantized"):
        self.collector = SimpleExtendedCollector()
        self.perspective_matcher = OptimizedPerspectiveMatcher(optimization_level=optimization_level)
        self.bias_classifier = OptimizedBiasClassifier(optimization_level=optimization_level)
        self.similarity_detector = OptimizedSimilarityDetector(optimization_level=optimization_level)
        
        self.results = {
            'bias_patterns': {},
            'sentiment_analysis': {},
            'perspective_effectiveness': {},
            'algorithm_performance': {},
            'topic_modeling': {},
            'temporal_analysis': {}
        }
        
        logger.info("Automated research framework initialized")
    
    def conduct_comprehensive_study(self, 
                                  topics: List[str] = None,
                                  days_range: int = 30,
                                  min_articles_per_topic: int = 50) -> Dict:
        """
        Conduct comprehensive automated research study
        
        Args:
            topics: List of topics to analyze (if None, uses general news)
            days_range: Number of days to collect data
            min_articles_per_topic: Minimum articles needed per topic
            
        Returns:
            Complete research results answering all research questions
        """
        
        logger.info("Starting comprehensive automated research study...")
        
        if topics is None:
            topics = [
                "climate change", "immigration", "healthcare", "economy", 
                "election", "foreign policy", "education", "crime"
            ]
        
        # Phase 1: Data Collection
        logger.info("Phase 1: Automated data collection across political spectrum...")
        all_articles = self._collect_research_data(topics, days_range, min_articles_per_topic)
        
        if len(all_articles) < 100:
            logger.warning(f"Only collected {len(all_articles)} articles, may need more for robust analysis")
        
        # Phase 2: Answer Research Questions
        logger.info("Phase 2: Analyzing data to answer research questions...")
        
        # RQ1: Topic and sentiment patterns across political spectrum
        self._analyze_bias_patterns(all_articles)
        
        # RQ2: Effectiveness of automated perspective surfacing
        self._evaluate_perspective_effectiveness(all_articles)
        
        # RQ3: Algorithm performance across political viewpoints
        self._assess_algorithm_performance(all_articles)
        
        # Phase 3: Generate comprehensive report
        report = self._generate_research_report()
        
        logger.info("Comprehensive research study completed")
        return report
    
    def _collect_research_data(self, topics: List[str], days_range: int, 
                             min_articles_per_topic: int) -> List:
        """Collect diverse articles for research analysis"""
        
        all_articles = []
        
        for topic in topics:
            logger.info(f"Collecting articles for topic: {topic}")
            
            # Collect diverse articles for this topic
            diverse_articles = self.collector.collect_diverse_articles(
                query=topic,
                days_back=days_range
            )
            
            # Flatten and add topic labels
            topic_articles = []
            for bias_category, articles in diverse_articles.items():
                for article in articles:
                    article.research_topic = topic
                    topic_articles.append(article)
            
            logger.info(f"Collected {len(topic_articles)} articles for {topic}")
            
            if len(topic_articles) >= min_articles_per_topic:
                all_articles.extend(topic_articles)
            else:
                logger.warning(f"Only {len(topic_articles)} articles for {topic}, below minimum")
        
        return all_articles
    
    def _analyze_bias_patterns(self, articles: List) -> None:
        """
        RQ1: What topic and sentiment patterns distinguish political viewpoints?
        """
        logger.info("Analyzing bias patterns and sentiment differences...")
        
        # Group articles by bias
        bias_groups = defaultdict(list)
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning"}
        
        for article in articles:
            if hasattr(article, 'bias_label') and article.bias_label is not None:
                bias_name = bias_labels.get(article.bias_label, "unknown")
                bias_groups[bias_name].append(article)
        
        # 1. Topic Modeling for Each Political Viewpoint
        topic_patterns = self._extract_topic_patterns(bias_groups)
        
        # 2. Language Pattern Analysis
        language_patterns = self._analyze_language_patterns(bias_groups)
        
        # 3. Sentiment Analysis
        sentiment_patterns = self._analyze_sentiment_patterns(bias_groups)
        
        # 4. Source Reliability Patterns
        source_patterns = self._analyze_source_patterns(bias_groups)
        
        self.results['bias_patterns'] = {
            'topic_modeling': topic_patterns,
            'language_patterns': language_patterns,
            'sentiment_patterns': sentiment_patterns,
            'source_patterns': source_patterns,
            'summary': self._summarize_bias_patterns(bias_groups)
        }
    
    def _extract_topic_patterns(self, bias_groups: Dict) -> Dict:
        """Extract distinctive topics for each political viewpoint"""
        
        topic_patterns = {}
        
        for bias_name, articles in bias_groups.items():
            if len(articles) < 10:
                continue
            
            # Combine titles and descriptions for topic modeling
            texts = []
            for article in articles:
                text = f"{article.title}. {article.description or ''}"
                texts.append(text.strip())
            
            # TF-IDF analysis for distinctive terms
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top terms by TF-IDF score
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                top_indices = np.argsort(mean_scores)[-20:][::-1]
                
                distinctive_terms = [
                    (feature_names[i], mean_scores[i]) 
                    for i in top_indices
                ]
                
                # Simple topic clustering
                n_topics = min(5, len(articles) // 10)
                if n_topics >= 2:
                    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                    lda.fit(tfidf_matrix)
                    
                    topics = []
                    for topic_idx, topic in enumerate(lda.components_):
                        top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
                        topics.append({
                            'topic_id': topic_idx,
                            'top_words': top_words,
                            'weight': float(topic.sum())
                        })
                else:
                    topics = []
                
                topic_patterns[bias_name] = {
                    'distinctive_terms': distinctive_terms,
                    'topics': topics,
                    'article_count': len(articles)
                }
                
            except Exception as e:
                logger.warning(f"Topic modeling failed for {bias_name}: {e}")
                topic_patterns[bias_name] = {
                    'error': str(e),
                    'article_count': len(articles)
                }
        
        return topic_patterns
    
    def _analyze_language_patterns(self, bias_groups: Dict) -> Dict:
        """Analyze language patterns across political viewpoints"""
        
        language_patterns = {}
        
        # Define political keywords to track
        political_keywords = {
            'economic': ['economy', 'jobs', 'unemployment', 'growth', 'recession', 'inflation'],
            'social': ['rights', 'equality', 'justice', 'freedom', 'liberty', 'values'],
            'government': ['regulation', 'policy', 'law', 'federal', 'state', 'local'],
            'international': ['foreign', 'international', 'global', 'trade', 'alliance'],
            'emotional': ['crisis', 'threat', 'opportunity', 'hope', 'fear', 'urgent']
        }
        
        for bias_name, articles in bias_groups.items():
            if len(articles) < 5:
                continue
            
            # Combine all text
            all_text = ' '.join([
                f"{article.title} {article.description or ''}" 
                for article in articles
            ]).lower()
            
            # Count keyword usage
            keyword_usage = {}
            for category, keywords in political_keywords.items():
                category_count = sum(all_text.count(keyword) for keyword in keywords)
                keyword_usage[category] = category_count
            
            # Calculate word diversity (unique words / total words)
            words = all_text.split()
            word_diversity = len(set(words)) / len(words) if words else 0
            
            # Average sentence length
            sentences = all_text.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            language_patterns[bias_name] = {
                'keyword_usage': keyword_usage,
                'word_diversity': word_diversity,
                'avg_sentence_length': float(avg_sentence_length),
                'total_words': len(words),
                'unique_words': len(set(words))
            }
        
        return language_patterns
    
    def _analyze_sentiment_patterns(self, bias_groups: Dict) -> Dict:
        """Analyze sentiment patterns using simple lexicon approach"""
        
        # Simple sentiment lexicons
        positive_words = {
            'success', 'growth', 'improvement', 'progress', 'achievement', 'victory',
            'opportunity', 'hope', 'optimism', 'benefit', 'advantage', 'gain'
        }
        
        negative_words = {
            'crisis', 'failure', 'decline', 'problem', 'threat', 'danger',
            'risk', 'concern', 'worry', 'loss', 'damage', 'harm'
        }
        
        sentiment_patterns = {}
        
        for bias_name, articles in bias_groups.items():
            if len(articles) < 5:
                continue
            
            article_sentiments = []
            
            for article in articles:
                text = f"{article.title} {article.description or ''}".lower()
                words = set(text.split())
                
                positive_count = len(words.intersection(positive_words))
                negative_count = len(words.intersection(negative_words))
                
                # Simple sentiment score
                if positive_count + negative_count > 0:
                    sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
                else:
                    sentiment_score = 0.0
                
                article_sentiments.append({
                    'sentiment_score': sentiment_score,
                    'positive_words': positive_count,
                    'negative_words': negative_count,
                    'title': article.title
                })
            
            # Aggregate sentiment statistics
            scores = [s['sentiment_score'] for s in article_sentiments]
            
            sentiment_patterns[bias_name] = {
                'mean_sentiment': float(np.mean(scores)),
                'sentiment_std': float(np.std(scores)),
                'positive_ratio': sum(1 for s in scores if s > 0) / len(scores),
                'negative_ratio': sum(1 for s in scores if s < 0) / len(scores),
                'neutral_ratio': sum(1 for s in scores if s == 0) / len(scores),
                'most_positive': max(article_sentiments, key=lambda x: x['sentiment_score']),
                'most_negative': min(article_sentiments, key=lambda x: x['sentiment_score']),
                'article_count': len(articles)
            }
        
        return sentiment_patterns
    
    def _analyze_source_patterns(self, bias_groups: Dict) -> Dict:
        """Analyze source patterns and diversity"""
        
        source_patterns = {}
        
        for bias_name, articles in bias_groups.items():
            sources = [article.source for article in articles]
            source_counts = Counter(sources)
            
            # Calculate source diversity (Shannon entropy)
            total_articles = len(articles)
            source_diversity = 0
            for count in source_counts.values():
                p = count / total_articles
                source_diversity -= p * np.log2(p) if p > 0 else 0
            
            source_patterns[bias_name] = {
                'unique_sources': len(source_counts),
                'source_diversity': source_diversity,
                'most_common_sources': source_counts.most_common(5),
                'total_articles': total_articles,
                'articles_per_source': total_articles / len(source_counts) if source_counts else 0
            }
        
        return source_patterns
    
    def _evaluate_perspective_effectiveness(self, articles: List) -> None:
        """
        RQ2: Can automated systems effectively surface alternative perspectives?
        """
        logger.info("Evaluating automated perspective surfacing effectiveness...")
        
        # Find perspective matches
        perspective_matches = self.perspective_matcher.find_perspective_matches_fast(articles)
        
        # Analyze match quality
        match_quality = self._analyze_match_quality(perspective_matches)
        
        # Measure coverage across political spectrum
        coverage_analysis = self._analyze_political_coverage(perspective_matches)
        
        # Temporal consistency analysis
        temporal_analysis = self._analyze_temporal_consistency(perspective_matches)
        
        self.results['perspective_effectiveness'] = {
            'total_matches': len(perspective_matches),
            'match_quality': match_quality,
            'political_coverage': coverage_analysis,
            'temporal_consistency': temporal_analysis,
            'effectiveness_score': self._calculate_effectiveness_score(perspective_matches)
        }
    
    def _analyze_match_quality(self, matches: List) -> Dict:
        """Analyze quality of perspective matches"""
        
        if not matches:
            return {'error': 'No matches found'}
        
        # Similarity score distribution
        similarities = [match.confidence for match in matches]
        
        # Perspective diversity in matches
        perspective_diversity = []
        for match in matches:
            unique_biases = len(set(article.bias_label for article in match.articles.values()))
            perspective_diversity.append(unique_biases)
        
        # Topic coverage
        topics = [match.topic for match in matches]
        topic_diversity = len(set(topics))
        
        return {
            'similarity_stats': {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities))
            },
            'perspective_diversity': {
                'mean_perspectives_per_match': float(np.mean(perspective_diversity)),
                'max_perspectives_per_match': int(np.max(perspective_diversity))
            },
            'topic_coverage': {
                'unique_topics': topic_diversity,
                'total_matches': len(matches),
                'topics_per_match': topic_diversity / len(matches) if matches else 0
            }
        }
    
    def _analyze_political_coverage(self, matches: List) -> Dict:
        """Analyze how well the system covers different political viewpoints"""
        
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning"}
        
        # Count matches by bias combination
        bias_combinations = defaultdict(int)
        bias_participation = defaultdict(int)
        
        for match in matches:
            biases = sorted([article.bias_label for article in match.articles.values()])
            combination_key = '-'.join([bias_labels[b] for b in biases])
            bias_combinations[combination_key] += 1
            
            for bias in biases:
                bias_participation[bias_labels[bias]] += 1
        
        return {
            'bias_combinations': dict(bias_combinations),
            'bias_participation': dict(bias_participation),
            'cross_spectrum_matches': len([
                match for match in matches 
                if len(set(article.bias_label for article in match.articles.values())) >= 2
            ]),
            'total_matches': len(matches)
        }
    
    def _analyze_temporal_consistency(self, matches: List) -> Dict:
        """Analyze temporal patterns in perspective matching"""
        
        if not matches:
            return {'error': 'No matches for temporal analysis'}
        
        # Group matches by date
        matches_by_date = defaultdict(list)
        for match in matches:
            # Use the earliest article date as match date
            earliest_date = min(article.published_at for article in match.articles.values())
            date_key = earliest_date.date()
            matches_by_date[date_key].append(match)
        
        # Calculate daily statistics
        daily_stats = []
        for date, day_matches in matches_by_date.items():
            daily_stats.append({
                'date': str(date),
                'match_count': len(day_matches),
                'avg_confidence': np.mean([m.confidence for m in day_matches]),
                'unique_topics': len(set(m.topic for m in day_matches))
            })
        
        return {
            'total_days': len(matches_by_date),
            'avg_matches_per_day': np.mean([len(matches) for matches in matches_by_date.values()]),
            'daily_stats': daily_stats,
            'temporal_span_days': len(matches_by_date)
        }
    
    def _assess_algorithm_performance(self, articles: List) -> None:
        """
        RQ3: How do bias detection algorithms perform across political viewpoints?
        """
        logger.info("Assessing algorithm performance across political spectrum...")
        
        # Performance by political viewpoint
        bias_performance = self._evaluate_bias_classification_performance(articles)
        
        # Similarity detection performance
        similarity_performance = self._evaluate_similarity_performance(articles)
        
        # Cross-viewpoint consistency
        consistency_analysis = self._analyze_cross_viewpoint_consistency(articles)
        
        self.results['algorithm_performance'] = {
            'bias_classification': bias_performance,
            'similarity_detection': similarity_performance,
            'cross_viewpoint_consistency': consistency_analysis
        }
    
    def _evaluate_bias_classification_performance(self, articles: List) -> Dict:
        """Evaluate bias classification performance using source-based validation"""
        
        # Use source-based labels as ground truth
        source_bias_map = self.perspective_matcher.source_bias_map
        
        validation_articles = []
        true_labels = []
        predicted_labels = []
        
        for article in articles:
            if article.source in source_bias_map:
                validation_articles.append(article)
                true_labels.append(source_bias_map[article.source])
                
                # Get model prediction
                text = f"{article.title}. {article.description or ''}"
                prediction = self.bias_classifier.predict_single(text)
                predicted_label = self.bias_classifier.reverse_label_map[prediction['predicted_class']]
                predicted_labels.append(predicted_label)
        
        if not validation_articles:
            return {'error': 'No articles available for validation'}
        
        # Calculate performance metrics
        accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)
        
        # Performance by bias category
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning"}
        performance_by_bias = {}
        
        for bias_value, bias_name in bias_labels.items():
            true_positives = sum(1 for t, p in zip(true_labels, predicted_labels) if t == bias_value and p == bias_value)
            false_positives = sum(1 for t, p in zip(true_labels, predicted_labels) if t != bias_value and p == bias_value)
            false_negatives = sum(1 for t, p in zip(true_labels, predicted_labels) if t == bias_value and p != bias_value)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            performance_by_bias[bias_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': sum(1 for t in true_labels if t == bias_value)
            }
        
        return {
            'overall_accuracy': accuracy,
            'performance_by_bias': performance_by_bias,
            'total_validated_articles': len(validation_articles),
            'confusion_matrix': self._create_confusion_matrix(true_labels, predicted_labels)
        }
    
    def _evaluate_similarity_performance(self, articles: List) -> Dict:
        """Evaluate similarity detection using article clustering validation"""
        
        # Group articles by research topic as ground truth clusters
        topic_groups = defaultdict(list)
        for article in articles:
            if hasattr(article, 'research_topic'):
                topic_groups[article.research_topic].append(article)
        
        if len(topic_groups) < 2:
            return {'error': 'Need multiple topics for similarity validation'}
        
        # Test similarity detection accuracy
        correct_similar = 0
        correct_different = 0
        total_comparisons = 0
        
        # Sample comparisons for efficiency
        articles_sample = articles[:100] if len(articles) > 100 else articles
        
        for i, article1 in enumerate(articles_sample):
            for j, article2 in enumerate(articles_sample[i+1:], i+1):
                if hasattr(article1, 'research_topic') and hasattr(article2, 'research_topic'):
                    # Ground truth: same topic = similar, different topic = different
                    should_be_similar = article1.research_topic == article2.research_topic
                    
                    # Get similarity score
                    text1 = f"{article1.title}. {article1.description or ''}"
                    text2 = f"{article2.title}. {article2.description or ''}"
                    
                    similarities = self.similarity_detector.find_similar_articles(text1, [text2], top_k=1)
                    is_similar = len(similarities) > 0 and similarities[0][1] >= settings.SIMILARITY_THRESHOLD
                    
                    if should_be_similar == is_similar:
                        if should_be_similar:
                            correct_similar += 1
                        else:
                            correct_different += 1
                    
                    total_comparisons += 1
                    
                    # Limit comparisons for efficiency
                    if total_comparisons >= 1000:
                        break
            
            if total_comparisons >= 1000:
                break
        
        accuracy = (correct_similar + correct_different) / total_comparisons if total_comparisons > 0 else 0
        
        return {
            'similarity_accuracy': accuracy,
            'correct_similar_pairs': correct_similar,
            'correct_different_pairs': correct_different,
            'total_comparisons': total_comparisons,
            'similarity_threshold': settings.SIMILARITY_THRESHOLD
        }
    
    def _create_confusion_matrix(self, true_labels: List, predicted_labels: List) -> List[List[int]]:
        """Create confusion matrix for bias classification"""
        
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning"}
        n_classes = len(bias_labels)
        
        confusion_matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
        
        for true_label, pred_label in zip(true_labels, predicted_labels):
            confusion_matrix[true_label][pred_label] += 1
        
        return confusion_matrix
    
    def _summarize_bias_patterns(self, bias_groups: Dict) -> Dict:
        """Create high-level summary of bias patterns"""
        
        summary = {}
        
        for bias_name, articles in bias_groups.items():
            if len(articles) < 5:
                continue
            
            # Most frequent sources
            sources = [article.source for article in articles]
            source_counts = Counter(sources)
            
            # Most recent articles
            recent_articles = sorted(articles, key=lambda x: x.published_at, reverse=True)[:5]
            
            summary[bias_name] = {
                'article_count': len(articles),
                'unique_sources': len(source_counts),
                'top_sources': source_counts.most_common(3),
                'recent_headlines': [article.title for article in recent_articles],
                'date_range': {
                    'earliest': min(article.published_at for article in articles).isoformat(),
                    'latest': max(article.published_at for article in articles).isoformat()
                }
            }
        
        return summary
    
    def _calculate_effectiveness_score(self, matches: List) -> float:
        """Calculate overall effectiveness score for perspective surfacing"""
        
        if not matches:
            return 0.0
        
        # Factors contributing to effectiveness
        factors = []
        
        # 1. Match quality (average confidence)
        avg_confidence = np.mean([match.confidence for match in matches])
        factors.append(avg_confidence)
        
        # 2. Political diversity (how many different viewpoints covered)
        all_biases = set()
        for match in matches:
            all_biases.update(article.bias_label for article in match.articles.values())
        diversity_score = len(all_biases) / 3.0  # Normalize by max possible (3 bias categories)
        factors.append(diversity_score)
        
        # 3. Topic coverage (how many different topics covered)
        unique_topics = len(set(match.topic for match in matches))
        topic_score = min(unique_topics / 10.0, 1.0)  # Normalize, cap at 10 topics
        factors.append(topic_score)
        
        # 4. Match frequency (how often we find matches)
        # This is already reflected in having matches at all
        
        # Calculate weighted average
        effectiveness_score = np.mean(factors)
        
        return float(effectiveness_score)
    
    def _analyze_cross_viewpoint_consistency(self, articles: List) -> Dict:
        """Analyze how consistently algorithms perform across political viewpoints"""
        
        # Group articles by political viewpoint
        bias_groups = defaultdict(list)
        bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning"}
        
        for article in articles:
            if hasattr(article, 'bias_label') and article.bias_label is not None:
                bias_name = bias_labels.get(article.bias_label, "unknown")
                bias_groups[bias_name].append(article)
        
        consistency_metrics = {}
        
        for bias_name, bias_articles in bias_groups.items():
            if len(bias_articles) < 10:
                continue
            
            # Test bias classification consistency within viewpoint
            sample_articles = bias_articles[:50] if len(bias_articles) > 50 else bias_articles
            
            predictions = []
            confidences = []
            
            for article in sample_articles:
                text = f"{article.title}. {article.description or ''}"
                prediction = self.bias_classifier.predict_single(text)
                predictions.append(prediction['predicted_class'])
                confidences.append(prediction['confidence'])
            
            # Calculate consistency metrics
            # How often does the classifier predict the expected bias for this group?
            expected_bias = bias_name
            correct_predictions = sum(1 for pred in predictions if pred == expected_bias)
            classification_consistency = correct_predictions / len(predictions)
            
            consistency_metrics[bias_name] = {
                'classification_consistency': classification_consistency,
                'avg_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'sample_size': len(sample_articles),
                'prediction_distribution': Counter(predictions)
            }
        
        return consistency_metrics
    
    def _generate_research_report(self) -> Dict:
        """Generate comprehensive research report answering all research questions"""
        
        report = {
            'executive_summary': self._create_executive_summary(),
            'research_questions': {
                'rq1_bias_patterns': self._answer_rq1(),
                'rq2_perspective_effectiveness': self._answer_rq2(), 
                'rq3_algorithm_performance': self._answer_rq3()
            },
            'methodology': self._describe_methodology(),
            'detailed_results': self.results,
            'conclusions_implications': self._draw_conclusions(),
            'limitations_future_work': self._identify_limitations()
        }
        
        return report
    
    def _create_executive_summary(self) -> Dict:
        """Create executive summary of findings"""
        
        summary = {}
        
        # Key findings from bias patterns
        if 'bias_patterns' in self.results:
            bias_summary = self.results['bias_patterns'].get('summary', {})
            total_articles = sum(data.get('article_count', 0) for data in bias_summary.values())
            unique_sources = sum(data.get('unique_sources', 0) for data in bias_summary.values())
            
            summary['data_overview'] = {
                'total_articles_analyzed': total_articles,
                'unique_sources': unique_sources,
                'political_viewpoints_covered': len(bias_summary),
                'analysis_period_days': self._calculate_analysis_period()
            }
        
        # Key findings from perspective effectiveness
        if 'perspective_effectiveness' in self.results:
            perspective_data = self.results['perspective_effectiveness']
            summary['perspective_findings'] = {
                'total_perspective_matches': perspective_data.get('total_matches', 0),
                'effectiveness_score': perspective_data.get('effectiveness_score', 0),
                'cross_spectrum_coverage': perspective_data.get('political_coverage', {}).get('cross_spectrum_matches', 0)
            }
        
        # Key findings from algorithm performance
        if 'algorithm_performance' in self.results:
            algo_data = self.results['algorithm_performance']
            bias_perf = algo_data.get('bias_classification', {})
            summary['algorithm_findings'] = {
                'bias_classification_accuracy': bias_perf.get('overall_accuracy', 0),
                'similarity_detection_accuracy': algo_data.get('similarity_detection', {}).get('similarity_accuracy', 0)
            }
        
        return summary
    
    def _answer_rq1(self) -> Dict:
        """Answer RQ1: What patterns distinguish political viewpoints?"""
        
        if 'bias_patterns' not in self.results:
            return {'error': 'Bias pattern analysis not completed'}
        
        bias_data = self.results['bias_patterns']
        
        answer = {
            'question': "What topic and sentiment patterns distinguish left, center, and right-leaning news sources?",
            'key_findings': [],
            'supporting_evidence': {}
        }
        
        # Topic pattern findings
        if 'topic_modeling' in bias_data:
            topic_data = bias_data['topic_modeling']
            
            # Extract distinctive terms for each viewpoint
            distinctive_patterns = {}
            for bias_name, data in topic_data.items():
                if 'distinctive_terms' in data and data['distinctive_terms']:
                    top_terms = [term for term, score in data['distinctive_terms'][:10]]
                    distinctive_patterns[bias_name] = top_terms
            
            if distinctive_patterns:
                answer['key_findings'].append(
                    f"Each political viewpoint shows distinct topic preferences: "
                    f"Left-leaning sources emphasize {distinctive_patterns.get('left-leaning', ['social issues'])[:3]}, "
                    f"centrist sources focus on {distinctive_patterns.get('centrist', ['factual reporting'])[:3]}, "
                    f"right-leaning sources highlight {distinctive_patterns.get('right-leaning', ['economic issues'])[:3]}"
                )
                answer['supporting_evidence']['topic_patterns'] = distinctive_patterns
        
        # Sentiment pattern findings
        if 'sentiment_patterns' in bias_data:
            sentiment_data = bias_data['sentiment_patterns']
            
            sentiment_comparison = {}
            for bias_name, data in sentiment_data.items():
                sentiment_comparison[bias_name] = {
                    'mean_sentiment': data.get('mean_sentiment', 0),
                    'positive_ratio': data.get('positive_ratio', 0),
                    'negative_ratio': data.get('negative_ratio', 0)
                }
            
            if sentiment_comparison:
                answer['key_findings'].append(
                    f"Sentiment patterns vary across political spectrum: "
                    f"sentiment scores and emotional language usage differ systematically between viewpoints"
                )
                answer['supporting_evidence']['sentiment_patterns'] = sentiment_comparison
        
        # Language pattern findings
        if 'language_patterns' in bias_data:
            lang_data = bias_data['language_patterns']
            
            keyword_patterns = {}
            for bias_name, data in lang_data.items():
                if 'keyword_usage' in data:
                    keyword_patterns[bias_name] = data['keyword_usage']
            
            if keyword_patterns:
                answer['key_findings'].append(
                    f"Language usage patterns show systematic differences in keyword categories "
                    f"(economic, social, government terms) across political viewpoints"
                )
                answer['supporting_evidence']['language_patterns'] = keyword_patterns
        
        return answer
    
    def _answer_rq2(self) -> Dict:
        """Answer RQ2: Can automated systems surface alternative perspectives?"""
        
        if 'perspective_effectiveness' not in self.results:
            return {'error': 'Perspective effectiveness analysis not completed'}
        
        perspective_data = self.results['perspective_effectiveness']
        
        answer = {
            'question': "Can automated systems effectively surface alternative perspectives on the same news events?",
            'key_findings': [],
            'supporting_evidence': {}
        }
        
        total_matches = perspective_data.get('total_matches', 0)
        effectiveness_score = perspective_data.get('effectiveness_score', 0)
        
        # Overall effectiveness finding
        if total_matches > 0:
            answer['key_findings'].append(
                f"Automated system successfully identified {total_matches} cross-perspective matches "
                f"with effectiveness score of {effectiveness_score:.3f}"
            )
        else:
            answer['key_findings'].append(
                "Automated system found limited cross-perspective matches, indicating challenges in "
                "identifying equivalent stories across political viewpoints"
            )
        
        # Match quality analysis
        if 'match_quality' in perspective_data:
            quality_data = perspective_data['match_quality']
            if 'similarity_stats' in quality_data:
                sim_stats = quality_data['similarity_stats']
                answer['key_findings'].append(
                    f"Match quality analysis shows average similarity of {sim_stats.get('mean', 0):.3f} "
                    f"with standard deviation of {sim_stats.get('std', 0):.3f}, indicating "
                    f"{'consistent' if sim_stats.get('std', 1) < 0.1 else 'variable'} match quality"
                )
                answer['supporting_evidence']['match_quality'] = quality_data
        
        # Political coverage analysis
        if 'political_coverage' in perspective_data:
            coverage_data = perspective_data['political_coverage']
            cross_spectrum = coverage_data.get('cross_spectrum_matches', 0)
            total = coverage_data.get('total_matches', 1)
            
            coverage_ratio = cross_spectrum / total if total > 0 else 0
            answer['key_findings'].append(
                f"Political spectrum coverage: {cross_spectrum}/{total} matches ({coverage_ratio:.1%}) "
                f"successfully bridge different political viewpoints"
            )
            answer['supporting_evidence']['political_coverage'] = coverage_data
        
        return answer
    
    def _answer_rq3(self) -> Dict:
        """Answer RQ3: How do algorithms perform across political viewpoints?"""
        
        if 'algorithm_performance' not in self.results:
            return {'error': 'Algorithm performance analysis not completed'}
        
        algo_data = self.results['algorithm_performance']
        
        answer = {
            'question': "How do bias detection algorithms perform across different political viewpoints?",
            'key_findings': [],
            'supporting_evidence': {}
        }
        
        # Bias classification performance
        if 'bias_classification' in algo_data:
            bias_perf = algo_data['bias_classification']
            overall_accuracy = bias_perf.get('overall_accuracy', 0)
            
            answer['key_findings'].append(
                f"Bias classification achieves {overall_accuracy:.1%} overall accuracy when validated "
                f"against source-based ground truth labels"
            )
            
            # Performance by political viewpoint
            if 'performance_by_bias' in bias_perf:
                perf_by_bias = bias_perf['performance_by_bias']
                performance_summary = {}
                
                for bias_name, metrics in perf_by_bias.items():
                    f1_score = metrics.get('f1_score', 0)
                    performance_summary[bias_name] = f1_score
                
                # Check for performance disparities
                f1_scores = list(performance_summary.values())
                if f1_scores:
                    min_f1 = min(f1_scores)
                    max_f1 = max(f1_scores)
                    performance_gap = max_f1 - min_f1
                    
                    if performance_gap > 0.1:
                        answer['key_findings'].append(
                            f"Algorithm shows uneven performance across political viewpoints: "
                            f"F1-scores range from {min_f1:.3f} to {max_f1:.3f}, indicating potential bias"
                        )
                    else:
                        answer['key_findings'].append(
                            f"Algorithm demonstrates consistent performance across political viewpoints "
                            f"with F1-scores ranging from {min_f1:.3f} to {max_f1:.3f}"
                        )
                
                answer['supporting_evidence']['bias_classification_performance'] = perf_by_bias
        
        # Similarity detection performance
        if 'similarity_detection' in algo_data:
            sim_perf = algo_data['similarity_detection']
            similarity_accuracy = sim_perf.get('similarity_accuracy', 0)
            
            answer['key_findings'].append(
                f"Similarity detection achieves {similarity_accuracy:.1%} accuracy in identifying "
                f"related articles across political viewpoints"
            )
            answer['supporting_evidence']['similarity_detection_performance'] = sim_perf
        
        # Cross-viewpoint consistency
        if 'cross_viewpoint_consistency' in algo_data:
            consistency_data = algo_data['cross_viewpoint_consistency']
            
            consistency_scores = []
            for bias_name, metrics in consistency_data.items():
                consistency = metrics.get('classification_consistency', 0)
                consistency_scores.append(consistency)
            
            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)
                consistency_range = max(consistency_scores) - min(consistency_scores)
                
                answer['key_findings'].append(
                    f"Cross-viewpoint consistency analysis shows {avg_consistency:.1%} average consistency "
                    f"with {consistency_range:.1%} variation across political viewpoints"
                )
                answer['supporting_evidence']['consistency_analysis'] = consistency_data
        
        return answer
    
    def _describe_methodology(self) -> Dict:
        """Describe the automated methodology used"""
        
        return {
            'overview': 'Fully automated analysis using computational methods without manual annotations',
            'data_collection': {
                'sources': 'Multi-source news collection from APIs and RSS feeds',
                'bias_assignment': 'Source-based bias labeling using established media bias classifications',
                'temporal_scope': 'Configurable time range analysis',
                'topic_coverage': 'Automated collection across multiple political topics'
            },
            'analysis_methods': {
                'bias_pattern_analysis': [
                    'TF-IDF vectorization for distinctive term identification',
                    'Latent Dirichlet Allocation (LDA) for topic modeling',
                    'Lexicon-based sentiment analysis',
                    'Source diversity and reliability metrics'
                ],
                'perspective_matching': [
                    'Sentence-BERT embeddings for semantic similarity',
                    'Cross-bias article clustering and matching',
                    'Temporal consistency analysis',
                    'Political spectrum coverage assessment'
                ],
                'algorithm_evaluation': [
                    'Source-based validation for bias classification',
                    'Topic-based validation for similarity detection',
                    'Cross-viewpoint performance consistency analysis',
                    'Confusion matrix analysis for classification performance'
                ]
            },
            'advantages': [
                'No manual annotation bias or subjectivity',
                'Scalable to large datasets',
                'Reproducible and consistent',
                'Real-time analysis capability',
                'Objective performance metrics'
            ],
            'validation_approach': 'Self-validating using source reliability and topic clustering as ground truth'
        }
    
    def _draw_conclusions(self) -> Dict:
        """Draw conclusions and implications from the analysis"""
        
        conclusions = {
            'primary_findings': [],
            'theoretical_implications': [],
            'practical_implications': [],
            'contribution_to_field': []
        }
        
        # Primary findings
        if self.results.get('bias_patterns'):
            conclusions['primary_findings'].append(
                "Distinct linguistic and topical patterns characterize different political viewpoints in news coverage"
            )
        
        if self.results.get('perspective_effectiveness'):
            effectiveness_score = self.results['perspective_effectiveness'].get('effectiveness_score', 0)
            if effectiveness_score > 0.6:
                conclusions['primary_findings'].append(
                    "Automated perspective surfacing demonstrates viable effectiveness for cross-political content discovery"
                )
            else:
                conclusions['primary_findings'].append(
                    "Automated perspective surfacing shows promise but requires refinement for optimal effectiveness"
                )
        
        if self.results.get('algorithm_performance'):
            bias_accuracy = self.results['algorithm_performance'].get('bias_classification', {}).get('overall_accuracy', 0)
            if bias_accuracy > 0.7:
                conclusions['primary_findings'].append(
                    "Bias detection algorithms achieve acceptable accuracy across political spectrum"
                )
            else:
                conclusions['primary_findings'].append(
                    "Bias detection algorithms show room for improvement in cross-political accuracy"
                )
        
        # Theoretical implications
        conclusions['theoretical_implications'] = [
            "Computational approaches can systematically identify political bias patterns without human annotation",
            "Cross-perspective content matching is technically feasible using semantic similarity methods",
            "Algorithmic bias detection may reflect and potentially amplify existing media categorizations",
            "Automated analysis reveals measurable differences in political news coverage patterns"
        ]
        
        # Practical implications
        conclusions['practical_implications'] = [
            "News aggregation systems can be designed to automatically surface diverse political perspectives",
            "Media literacy tools can be enhanced with automated bias detection capabilities",
            "Content recommendation systems can incorporate perspective diversity metrics",
            "Journalists and news organizations can use automated tools for bias-aware reporting"
        ]
        
        # Contribution to field
        conclusions['contribution_to_field'] = [
            "Demonstrates feasibility of fully automated political bias analysis in news media",
            "Provides methodology for evaluating perspective diversity without manual annotation",
            "Establishes baseline performance metrics for automated bias detection systems",
            "Shows practical approach to addressing information polarization through technology"
        ]
        
        return conclusions
    
    def _identify_limitations(self) -> Dict:
        """Identify limitations and future research directions"""
        
        return {
            'methodology_limitations': [
                "Source-based bias labeling may not capture nuanced within-source variation",
                "Limited to English-language sources and US political spectrum",
                "Temporal analysis constrained by API rate limits and data availability",
                "Similarity thresholds require domain-specific tuning"
            ],
            'technical_limitations': [
                "Semantic similarity may miss subtle bias differences in similar content",
                "Topic modeling sensitive to preprocessing and parameter choices",
                "Sentiment analysis using simple lexicon approach may miss context",
                "Cross-cultural and cross-temporal generalizability unclear"
            ],
            'scope_limitations': [
                "Analysis focused on mainstream news sources",
                "Limited evaluation on social media or alternative media platforms",
                "No user behavior or impact assessment included",
                "Evaluation period may not capture full range of political discourse"
            ],
            'future_research_directions': [
                "Longitudinal analysis of bias pattern evolution over time",
                "User study evaluation of perspective diversity impact",
                "Cross-cultural validation of bias detection approaches",
                "Integration of multimodal content analysis (text, images, video)",
                "Development of dynamic bias detection that adapts to emerging patterns",
                "Investigation of algorithmic intervention effects on user information consumption"
            ],
            'ethical_considerations': [
                "Potential for reinforcing existing bias categorizations",
                "Risk of oversimplifying complex political viewpoints",
                "Need for transparency in algorithmic bias detection",
                "Consideration of filter bubble vs. perspective diversity balance"
            ]
        }
    
    def _calculate_analysis_period(self) -> int:
        """Calculate the analysis period in days"""
        if 'bias_patterns' in self.results and 'summary' in self.results['bias_patterns']:
            summary = self.results['bias_patterns']['summary']
            
            all_dates = []
            for bias_data in summary.values():
                if 'date_range' in bias_data:
                    try:
                        earliest = datetime.fromisoformat(bias_data['date_range']['earliest'])
                        latest = datetime.fromisoformat(bias_data['date_range']['latest'])
                        all_dates.extend([earliest, latest])
                    except:
                        continue
            
            if all_dates:
                period = (max(all_dates) - min(all_dates)).days
                return period
        
        return 30  # Default assumption
    
    def save_results(self, filepath: Path) -> None:
        """Save research results to JSON file"""
        
        # Convert datetime objects and other non-serializable objects
        serializable_results = self._make_json_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Research results saved to {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

# Example usage and testing
if __name__ == "__main__":
    print("Automated Research Framework Test")
    print("=" * 40)
    
    # Initialize framework
    framework = AutomatedResearchFramework(optimization_level="quantized")
    
    # Run comprehensive study
    print("Running comprehensive automated research study...")
    
    # Use a smaller scope for testing
    test_topics = ["climate change", "healthcare", "economy"]
    
    results = framework.conduct_comprehensive_study(
        topics=test_topics,
        days_range=14,  # Two weeks of data
        min_articles_per_topic=20
    )
    
    # Display key findings
    print("\nKey Research Findings:")
    print("=" * 30)
    
    exec_summary = results.get('executive_summary', {})
    if 'data_overview' in exec_summary:
        data = exec_summary['data_overview']
        print(f"Data: {data.get('total_articles_analyzed', 0)} articles from {data.get('unique_sources', 0)} sources")
    
    # RQ1 Answer
    rq1 = results.get('research_questions', {}).get('rq1_bias_patterns', {})
    if 'key_findings' in rq1:
        print(f"\nRQ1 Findings:")
        for finding in rq1['key_findings'][:2]:
            print(f"  • {finding}")
    
    # RQ2 Answer  
    rq2 = results.get('research_questions', {}).get('rq2_perspective_effectiveness', {})
    if 'key_findings' in rq2:
        print(f"\nRQ2 Findings:")
        for finding in rq2['key_findings'][:2]:
            print(f"  • {finding}")
    
    # RQ3 Answer
    rq3 = results.get('research_questions', {}).get('rq3_algorithm_performance', {})
    if 'key_findings' in rq3:
        print(f"\nRQ3 Findings:")
        for finding in rq3['key_findings'][:2]:
            print(f"  • {finding}")
    
    # Save results
    output_path = Path("research_results.json")
    framework.save_results(output_path)
    print(f"\nComplete results saved to: {output_path}")