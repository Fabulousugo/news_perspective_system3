# src/models/news_browser.py - Better user-focused news browsing system

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict

from .similarity_detector import SimilarityDetector
from ..data_collection.news_apis import Article
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ArticleWithPerspectives:
    """Article with information about alternative perspectives available"""
    article: Article
    related_articles: List[Tuple[Article, float]]  # (article, similarity_score)
    perspective_count: int
    topic_cluster: str
    
class NewsBrowser:
    """
    User-friendly news browsing system focused on article discovery first,
    then perspective exploration
    """
    
    def __init__(self):
        self.similarity_detector = SimilarityDetector()
        
        # Much more relaxed similarity thresholds for better recall
        self.loose_threshold = 0.65   # For finding any related articles
        self.tight_threshold = 0.75   # For high-confidence matches
        
        # Source bias mapping
        self.source_bias_map = {
            # Left-leaning (0)
            'CNN': 0, 'cnn': 0, 'The Guardian': 0, 'MSNBC': 0, 'NPR': 0,
            'the-guardian-uk': 0, 'msnbc': 0, 'npr': 0,
            
            # Centrist (1) 
            'Reuters': 1, 'Associated Press': 1, 'BBC News': 1,
            'reuters': 1, 'associated-press': 1, 'bbc-news': 1, 'AP': 1,
            
            # Right-leaning (2)
            'Fox News': 2, 'New York Post': 2, 'Wall Street Journal': 2,
            'fox-news': 2, 'new-york-post': 2, 'the-wall-street-journal': 2,
            'Washington Examiner': 2, 'washington-examiner': 2
        }
        
        logger.info("✅ News browser initialized with relaxed similarity thresholds")
    
    def assign_bias_labels(self, articles: List[Article]) -> List[Article]:
        """Assign bias labels based on source"""
        labeled_articles = []
        
        for article in articles:
            bias_label = article.bias_label
            
            if bias_label is None:
                bias_label = self.source_bias_map.get(article.source)
                if bias_label is None:
                    # Try variations
                    source_lower = article.source.lower().replace(' ', '-')
                    bias_label = self.source_bias_map.get(source_lower, 1)  # Default centrist
            
            labeled_article = Article(
                title=article.title,
                content=article.content,
                url=article.url,
                source=article.source,
                published_at=article.published_at,
                author=article.author,
                description=article.description,
                bias_label=bias_label
            )
            labeled_articles.append(labeled_article)
        
        return labeled_articles
    
    def browse_articles(self, articles: List[Article], sort_by: str = "recent") -> List[ArticleWithPerspectives]:
        """
        Main browsing interface - shows articles with perspective information
        
        Args:
            articles: List of articles to browse
            sort_by: "recent", "diverse", "popular"
            
        Returns:
            List of articles enriched with perspective information
        """
        logger.info(f"🔍 Analyzing {len(articles)} articles for perspective discovery...")
        
        # Assign bias labels
        labeled_articles = self.assign_bias_labels(articles)
        
        # Find all potential perspective matches with relaxed thresholds
        perspective_map = self._build_perspective_map(labeled_articles)
        
        # Create browseable articles
        browseable_articles = []
        processed_urls = set()
        
        for article in labeled_articles:
            if article.url in processed_urls:
                continue
                
            # Get related articles for this one
            related = perspective_map.get(article.url, [])
            
            # Filter out different perspectives
            different_bias_related = []
            article_bias = article.bias_label
            
            for related_article, similarity in related:
                if related_article.bias_label != article_bias:
                    different_bias_related.append((related_article, similarity))
            
            # Create browseable article
            browseable = ArticleWithPerspectives(
                article=article,
                related_articles=different_bias_related,
                perspective_count=len(different_bias_related),
                topic_cluster=self._extract_topic_simple([article])
            )
            
            browseable_articles.append(browseable)
            processed_urls.add(article.url)
        
        # Sort articles
        if sort_by == "diverse":
            browseable_articles.sort(key=lambda x: x.perspective_count, reverse=True)
        elif sort_by == "recent":
            browseable_articles.sort(key=lambda x: x.article.published_at, reverse=True)
        
        logger.info(f"✅ Found {len(browseable_articles)} articles")
        perspective_counts = [a.perspective_count for a in browseable_articles]
        if perspective_counts:
            logger.info(f"📊 Perspective stats: avg={np.mean(perspective_counts):.1f}, max={max(perspective_counts)}")
        
        return browseable_articles
    
    def _build_perspective_map(self, articles: List[Article]) -> Dict[str, List[Tuple[Article, float]]]:
        """Build a map of article URL -> related articles from other perspectives"""
        perspective_map = defaultdict(list)
        
        # Group articles by bias
        bias_groups = defaultdict(list)
        bias_labels = {0: "left", 1: "center", 2: "right"}
        
        for article in articles:
            bias_category = bias_labels.get(article.bias_label, "center")
            bias_groups[bias_category].append(article)
        
        logger.info(f"📊 Bias distribution: {[(k, len(v)) for k, v in bias_groups.items()]}")
        
        # Compare articles across different bias groups
        bias_list = list(bias_groups.keys())
        
        for i, bias1 in enumerate(bias_list):
            for j, bias2 in enumerate(bias_list):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue
                
                articles1 = bias_groups[bias1]
                articles2 = bias_groups[bias2]
                
                logger.info(f"🔄 Comparing {bias1} vs {bias2}: {len(articles1)} vs {len(articles2)} articles")
                
                # Find similarities between groups
                for article1 in articles1:
                    text1 = self._prepare_text_for_similarity(article1)
                    
                    # Prepare all texts from the other group
                    candidate_texts = []
                    for article2 in articles2:
                        candidate_texts.append(self._prepare_text_for_similarity(article2))
                    
                    if not candidate_texts:
                        continue
                    
                    try:
                        # Temporarily disable progress bars for cleaner output
                        import os
                        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                        
                        # Find similar articles with relaxed threshold
                        similarities = self.similarity_detector.find_similar_articles(
                            text1, candidate_texts, top_k=10  # Get more candidates
                        )
                        
                        # Add all matches above loose threshold
                        for idx, similarity in similarities:
                            if similarity >= self.loose_threshold:  # Much more relaxed
                                related_article = articles2[idx]
                                perspective_map[article1.url].append((related_article, similarity))
                                perspective_map[related_article.url].append((article1, similarity))
                                
                    except Exception as e:
                        logger.warning(f"Similarity calculation failed: {e}")
                        continue
        
        return perspective_map
    
    def _prepare_text_for_similarity(self, article: Article) -> str:
        """Prepare article text for similarity comparison"""
        # Use title + description for better matching
        text_parts = [article.title]
        
        if article.description:
            text_parts.append(article.description)
        
        # Use first part of content if available and description is short
        if article.content and (not article.description or len(article.description) < 100):
            # Take first few sentences of content
            content_preview = ' '.join(article.content.split()[:50])
            text_parts.append(content_preview)
        
        full_text = ' '.join(text_parts)
        return full_text.strip()
    
    def _extract_topic_simple(self, articles: List[Article]) -> str:
        """Extract topic from article titles"""
        if not articles:
            return "General"
        
        # Get words from titles
        all_words = []
        for article in articles:
            words = article.title.lower().split()
            all_words.extend(words)
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'this', 'that'}
        meaningful_words = [w for w in all_words if len(w) > 3 and w not in stop_words]
        
        if meaningful_words:
            # Return most common word
            word_counts = {}
            for word in meaningful_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            if word_counts:
                most_common = max(word_counts.items(), key=lambda x: x[1])
                return most_common[0].title()
        
        return "General"
    
    def get_article_details(self, article_url: str, all_articles: List[Article]) -> Optional[Dict]:
        """Get detailed view of a specific article with its perspectives"""
        # Find the article
        target_article = None
        for article in all_articles:
            if article.url == article_url:
                target_article = article
                break
        
        if not target_article:
            return None
        
        # Get browseable articles to find perspectives
        browseable_articles = self.browse_articles(all_articles)
        
        # Find the browseable version of this article
        for browseable in browseable_articles:
            if browseable.article.url == article_url:
                return {
                    'article': asdict(browseable.article),
                    'perspectives': [
                        {
                            'article': asdict(related[0]),
                            'similarity': related[1],
                            'bias_category': {0: 'left-leaning', 1: 'centrist', 2: 'right-leaning'}[related[0].bias_label]
                        }
                        for related in browseable.related_articles
                    ],
                    'perspective_count': browseable.perspective_count,
                    'topic': browseable.topic_cluster
                }
        
        return None
    
    def search_articles(self, articles: List[Article], query: str) -> List[ArticleWithPerspectives]:
        """Search articles by keyword"""
        query_lower = query.lower()
        
        # Filter articles that match the query
        matching_articles = []
        for article in articles:
            if (query_lower in article.title.lower() or 
                (article.description and query_lower in article.description.lower())):
                matching_articles.append(article)
        
        # Return browseable version
        return self.browse_articles(matching_articles)
    
    def get_statistics(self, browseable_articles: List[ArticleWithPerspectives]) -> Dict:
        """Get statistics about perspective coverage"""
        total_articles = len(browseable_articles)
        articles_with_perspectives = len([a for a in browseable_articles if a.perspective_count > 0])
        
        perspective_counts = [a.perspective_count for a in browseable_articles]
        avg_perspectives = np.mean(perspective_counts) if perspective_counts else 0
        max_perspectives = max(perspective_counts) if perspective_counts else 0
        
        # Bias distribution
        bias_distribution = defaultdict(int)
        bias_labels = {0: 'left-leaning', 1: 'centrist', 2: 'right-leaning'}
        
        for browseable in browseable_articles:
            bias_name = bias_labels.get(browseable.article.bias_label, 'unknown')
            bias_distribution[bias_name] += 1
        
        return {
            'total_articles': total_articles,
            'articles_with_perspectives': articles_with_perspectives,
            'perspective_coverage': articles_with_perspectives / total_articles if total_articles > 0 else 0,
            'average_perspectives_per_article': avg_perspectives,
            'max_perspectives_found': max_perspectives,
            'bias_distribution': dict(bias_distribution),
            'similarity_thresholds': {
                'loose': self.loose_threshold,
                'tight': self.tight_threshold
            }
        }

# Example usage
if __name__ == "__main__":
    browser = NewsBrowser()
    print("✅ News browser ready for testing")