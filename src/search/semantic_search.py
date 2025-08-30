# src/search/semantic_search.py - Enhanced semantic search with query expansion

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from sentence_transformers import SentenceTransformer
import logging
import re
import os,sys
from dataclasses import dataclass
from collections import defaultdict

from ..data_collection.news_apis import Article
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Enhanced search result with relevance scoring"""
    article: Article
    relevance_score: float
    matched_terms: List[str]
    semantic_similarity: float
    query_coverage: float

class QueryExpander:
    """Handles query expansion and synonym recognition"""
    
    def __init__(self):
        # Common synonyms and expansions for news topics
        self.synonyms = {
            # Technology
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
            'artificial intelligence': ['ai', 'machine learning', 'deep learning', 'neural networks'],
            'ml': ['machine learning', 'artificial intelligence', 'ai'],
            'tech': ['technology', 'technological', 'innovation'],
            'cryptocurrency': ['crypto', 'bitcoin', 'blockchain', 'digital currency'],
            'crypto': ['cryptocurrency', 'bitcoin', 'blockchain', 'digital currency'],
            
            # Politics
            'biden': ['joe biden', 'president biden', 'biden administration'],
            'trump': ['donald trump', 'former president trump', 'trump administration'],
            'gop': ['republican', 'republican party', 'republicans'],
            'democrat': ['democratic', 'democratic party', 'democrats'],
            'election': ['elections', 'voting', 'ballot', 'electoral'],
            'healthcare': ['health care', 'medical care', 'obamacare', 'affordable care act'],
            
            # Environment
            'climate change': ['global warming', 'climate crisis', 'environmental'],
            'global warming': ['climate change', 'climate crisis', 'environmental'],
            'renewable': ['renewable energy', 'clean energy', 'solar', 'wind power'],
            
            # Economy
            'economy': ['economic', 'economics', 'gdp', 'recession', 'inflation'],
            'jobs': ['employment', 'unemployment', 'labor', 'workforce'],
            'inflation': ['price increases', 'cost of living', 'economic'],
            
            # International
            'ukraine': ['ukrainian', 'kyiv', 'zelensky', 'russia ukraine'],
            'russia': ['russian', 'putin', 'moscow', 'kremlin'],
            'china': ['chinese', 'beijing', 'xi jinping', 'ccp'],
        }
        
        # Exclusion patterns - if searching for A, exclude articles primarily about B
        self.exclusions = {
            'biden': ['trump', 'donald trump', 'former president trump'],
            'trump': ['biden', 'joe biden', 'president biden'],
            'republican': ['democrat', 'democratic'],
            'democrat': ['republican', 'gop'],
            'climate change': ['climate denial', 'climate hoax'],
        }
        
        # Weighted terms - some synonyms are more important
        self.term_weights = {
            # Exact matches get highest weight
            'exact': 1.0,
            # Close synonyms get high weight
            'synonym': 0.9,
            # Related terms get medium weight
            'related': 0.7,
            # Broad terms get lower weight
            'broad': 0.5
        }
    
    def expand_query(self, query: str) -> Dict[str, float]:
        """
        Expand query with synonyms and related terms
        
        Returns:
            Dictionary of terms with weights
        """
        query_lower = query.lower().strip()
        expanded_terms = {query_lower: self.term_weights['exact']}
        
        # Add exact synonyms
        if query_lower in self.synonyms:
            for synonym in self.synonyms[query_lower]:
                expanded_terms[synonym] = self.term_weights['synonym']
        
        # Add partial matches for compound queries
        query_words = query_lower.split()
        for word in query_words:
            if word in self.synonyms:
                for synonym in self.synonyms[word]:
                    expanded_terms[synonym] = self.term_weights['related']
        
        logger.info(f"🔍 Query '{query}' expanded to {len(expanded_terms)} terms")
        return expanded_terms
    
    def get_exclusions(self, query: str) -> List[str]:
        """Get terms that should reduce relevance for this query"""
        query_lower = query.lower().strip()
        return self.exclusions.get(query_lower, [])

class EnhancedSemanticSearch:
    """
    Enhanced semantic search with query expansion and relevance filtering
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.similarity_model = SentenceTransformer(model_name)
        self.query_expander = QueryExpander()
        
        # Search thresholds
        self.semantic_threshold = 0.3  # Lower threshold for semantic similarity
        self.relevance_threshold = 0.4  # Minimum relevance score
        self.exclusion_penalty = 0.3   # Penalty for exclusion terms
        
        logger.info("✅ Enhanced semantic search initialized")
    
    def search_articles(self, articles: List[Article], query: str, 
                       top_k: int = 50) -> List[SearchResult]:
        """
        Enhanced semantic search with query expansion and relevance filtering
        
        Args:
            articles: List of articles to search
            query: Search query
            top_k: Maximum number of results
            
        Returns:
            List of SearchResult objects ordered by relevance
        """
        if not articles or not query.strip():
            return []
        
        logger.info(f"🔍 Semantic search: '{query}' across {len(articles)} articles")
        
        # Step 1: Expand query with synonyms
        expanded_terms = self.query_expander.expand_query(query)
        exclusion_terms = self.query_expander.get_exclusions(query)
        
        # Step 2: Prepare article texts and compute embeddings
        article_texts = []
        for article in articles:
            # Combine title, description, and content snippet for search
            text_parts = [article.title]
            if article.description:
                text_parts.append(article.description)
            if article.content:
                # Add first 200 characters of content
                content_snippet = article.content[:200]
                text_parts.append(content_snippet)
            
            full_text = ' '.join(text_parts)
            article_texts.append(full_text)
        
        # Step 3: Compute semantic similarities
        query_embedding = self.similarity_model.encode([query])
        article_embeddings = self.similarity_model.encode(article_texts)
        
        semantic_similarities = np.dot(article_embeddings, query_embedding.T).flatten()
        
        # Step 4: Compute comprehensive relevance scores
        search_results = []
        
        for i, article in enumerate(articles):
            # Get semantic similarity
            semantic_sim = float(semantic_similarities[i])
            
            # Skip articles with very low semantic similarity
            if semantic_sim < self.semantic_threshold:
                continue
            
            # Compute keyword relevance
            keyword_score, matched_terms = self._compute_keyword_relevance(
                article_texts[i], expanded_terms
            )
            
            # Compute query coverage
            query_coverage = self._compute_query_coverage(
                article_texts[i], query, expanded_terms
            )
            
            # Apply exclusion penalty
            exclusion_penalty = self._compute_exclusion_penalty(
                article_texts[i], exclusion_terms
            )
            
            # Combine scores
            relevance_score = self._combine_scores(
                semantic_sim, keyword_score, query_coverage, exclusion_penalty
            )
            
            # Filter by minimum relevance
            if relevance_score >= self.relevance_threshold:
                search_result = SearchResult(
                    article=article,
                    relevance_score=relevance_score,
                    matched_terms=matched_terms,
                    semantic_similarity=semantic_sim,
                    query_coverage=query_coverage
                )
                search_results.append(search_result)
        
        # Step 5: Sort by relevance and return top results
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"✅ Found {len(search_results)} relevant articles")
        return search_results[:top_k]
    
    def _compute_keyword_relevance(self, text: str, expanded_terms: Dict[str, float]) -> Tuple[float, List[str]]:
        """Compute keyword-based relevance score"""
        text_lower = text.lower()
        matched_terms = []
        total_score = 0.0
        
        for term, weight in expanded_terms.items():
            # Count occurrences (with some normalization for length)
            count = text_lower.count(term.lower())
            if count > 0:
                matched_terms.append(term)
                # Logarithmic scaling to prevent over-weighting repeated terms
                term_score = weight * (1 + np.log(count))
                total_score += term_score
        
        # Normalize by text length (roughly)
        text_length_factor = min(len(text) / 1000, 1.0)  # Cap at 1000 chars
        normalized_score = total_score * text_length_factor
        
        return min(normalized_score, 1.0), matched_terms
    
    def _compute_query_coverage(self, text: str, original_query: str, 
                              expanded_terms: Dict[str, float]) -> float:
        """Compute how well the article covers the query terms"""
        query_words = set(original_query.lower().split())
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Direct word overlap
        direct_overlap = len(query_words.intersection(text_words))
        direct_coverage = direct_overlap / len(query_words) if query_words else 0
        
        # Expanded term coverage
        expanded_matches = 0
        for term in expanded_terms:
            if term.lower() in text.lower():
                expanded_matches += 1
        
        expanded_coverage = expanded_matches / len(expanded_terms) if expanded_terms else 0
        
        # Combine coverages
        total_coverage = 0.7 * direct_coverage + 0.3 * expanded_coverage
        return min(total_coverage, 1.0)
    
    def _compute_exclusion_penalty(self, text: str, exclusion_terms: List[str]) -> float:
        """Compute penalty for exclusion terms"""
        if not exclusion_terms:
            return 0.0
        
        text_lower = text.lower()
        penalty = 0.0
        
        for term in exclusion_terms:
            count = text_lower.count(term.lower())
            if count > 0:
                # More mentions = higher penalty
                penalty += self.exclusion_penalty * (1 + np.log(count))
        
        return min(penalty, 0.8)  # Cap penalty at 0.8
    
    def _combine_scores(self, semantic_sim: float, keyword_score: float, 
                       query_coverage: float, exclusion_penalty: float) -> float:
        """Combine different relevance signals into final score"""
        # Weighted combination
        combined_score = (
            0.4 * semantic_sim +      # Semantic understanding
            0.3 * keyword_score +     # Keyword matching
            0.3 * query_coverage      # Query coverage
        )
        
        # Apply exclusion penalty
        final_score = combined_score - exclusion_penalty
        
        return max(final_score, 0.0)  # Ensure non-negative

class SmartQueryProcessor:
    """Processes and cleans queries for better search results"""
    
    def __init__(self):
        # Common query patterns and their standardized forms
        self.query_patterns = {
            # Technology patterns
            r'\b(ai|a\.i\.)\b': 'artificial intelligence',
            r'\bmachine learning\b': 'artificial intelligence',
            r'\bcrypto(?:currency)?\b': 'cryptocurrency',
            
            # Political patterns  
            r'\bpres(?:ident)?\s+biden\b': 'biden',
            r'\bpres(?:ident)?\s+trump\b': 'trump',
            r'\bformer\s+pres(?:ident)?\s+trump\b': 'trump',
            r'\bgop\b': 'republican',
            
            # Issue patterns
            r'\bclimate\s+change\b': 'climate change',
            r'\bglobal\s+warming\b': 'climate change',
            r'\bhealth\s+care\b': 'healthcare',
        }
    
    def process_query(self, query: str) -> str:
        """Process and standardize query"""
        processed_query = query.strip().lower()
        
        # Apply pattern replacements
        for pattern, replacement in self.query_patterns.items():
            processed_query = re.sub(pattern, replacement, processed_query, flags=re.IGNORECASE)
        
        return processed_query.strip()

# Integration with existing system
def enhance_news_browser_search():
    """Enhance the existing news browser with semantic search"""
    
    # This will be integrated into the news browser
    pass

# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced search
    print("🧪 Testing Enhanced Semantic Search")
    print("=" * 40)
    
    # Create test articles
    from datetime import datetime
    test_articles = [
        Article("AI Revolution in Healthcare", "Artificial intelligence transforms medical diagnosis", "url1", "CNN", datetime.now()),
        Article("Machine Learning Breakthrough", "New AI system achieves human-level performance", "url2", "Reuters", datetime.now()),
        Article("Biden Signs AI Executive Order", "President Biden announces new artificial intelligence regulations", "url3", "AP", datetime.now()),
        Article("Trump Criticizes AI Policies", "Former president Trump opposes Biden's AI regulations", "url4", "Fox News", datetime.now()),
        Article("Climate Change Report", "Scientists warn of urgent climate action needed", "url5", "Guardian", datetime.now()),
        Article("Tech Innovation Summit", "Leaders discuss future of technology and innovation", "url6", "WSJ", datetime.now()),
        Article("Biden Climate Policy", "President announces new environmental initiatives", "url7", "NPR", datetime.now()),
    ]
    
    # Initialize search
    search_engine = EnhancedSemanticSearch()
    
    # Test queries that were problematic
    test_queries = ["AI", "artificial intelligence", "Biden", "climate change"]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        results = search_engine.search_articles(test_articles, query, top_k=5)
        
        print(f"✅ Found {len(results)} relevant results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.article.title}")
            print(f"     Relevance: {result.relevance_score:.3f}")
            print(f"     Semantic: {result.semantic_similarity:.3f}")
            print(f"     Matched: {', '.join(result.matched_terms[:3])}")