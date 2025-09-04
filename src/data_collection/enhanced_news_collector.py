# src/data_collection/enhanced_news_collector.py

import requests
import feedparser
import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import yaml
from abc import ABC, abstractmethod
import re
from urllib.parse import urlparse

from .news_apis import NewsAPIBase, Article
from config.settings import settings

logger = logging.getLogger(__name__)

class RSSCollector:
    """RSS feed collector for sources without API access"""
    
    def __init__(self, rate_limit_delay: float = 2.0):
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Rate limiting for RSS requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def fetch_rss_articles(self, rss_url: str, source_name: str, 
                          bias_score: int, days_back: int = 7) -> List[Article]:
        """
        Fetch articles from RSS feed
        
        Args:
            rss_url: RSS feed URL
            source_name: Name of the news source
            bias_score: Political bias score (0=left, 1=center, 2=right)
            days_back: Number of days to look back
            
        Returns:
            List of Article objects
        """
        self._rate_limit()
        
        try:
            logger.info(f"Fetching RSS from {source_name}: {rss_url}")
            
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            
            if feed.bozo and feed.bozo_exception:
                logger.warning(f"RSS feed issues for {source_name}: {feed.bozo_exception}")
            
            articles = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for entry in feed.entries:
                try:
                    # Parse publication date
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    else:
                        # Fallback to current time
                        pub_date = datetime.now()
                    
                    # Skip old articles
                    if pub_date < cutoff_date:
                        continue
                    
                    # Extract content
                    content = ""
                    if hasattr(entry, 'content') and entry.content:
                        content = entry.content[0].value
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    elif hasattr(entry, 'description'):
                        content = entry.description
                    
                    # Clean HTML from content
                    content = self._clean_html(content)
                    
                    # Extract description/summary
                    description = ""
                    if hasattr(entry, 'summary'):
                        description = self._clean_html(entry.summary)
                    
                    # Skip if too short
                    if len(content) < settings.MIN_ARTICLE_LENGTH and len(description) < 50:
                        continue
                    
                    # Create article
                    article = Article(
                        title=entry.title,
                        content=content,
                        url=entry.link,
                        source=source_name,
                        published_at=pub_date,
                        author=getattr(entry, 'author', None),
                        description=description,
                        bias_label=bias_score
                    )
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse RSS entry from {source_name}: {e}")
                    continue
            
            logger.info(f"Collected {len(articles)} articles from {source_name} RSS")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch RSS from {source_name}: {e}")
            return []
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        if not text:
            return ""
        
        # Remove HTML tags
        clean = re.sub('<.*?>', '', text)
        
        # Replace HTML entities
        clean = clean.replace('&amp;', '&')
        clean = clean.replace('&lt;', '<')
        clean = clean.replace('&gt;', '>')
        clean = clean.replace('&quot;', '"')
        clean = clean.replace('&#39;', "'")
        clean = clean.replace('&nbsp;', ' ')
        
        # Clean up whitespace
        clean = ' '.join(clean.split())
        
        return clean.strip()

class EnhancedNewsCollector:
    """
    Enhanced news collector supporting multiple APIs and RSS feeds
    """
    
    def __init__(self, config_file: str = "config/expanded_news_sources.yaml"):
        self.config_file = config_file
        self.apis = {}
        self.rss_collector = RSSCollector()
        self.source_configs = self._load_source_configs()
        self._initialize_apis()
        
        logger.info("Enhanced news collector initialized")
    
    def _load_source_configs(self) -> Dict:
        """Load source configurations from YAML file"""
        try:
            config_path = settings.PROJECT_ROOT / self.config_file
            if not config_path.exists():
                # Fallback to basic config
                config_path = settings.PROJECT_ROOT / "config" / "news_sources.yaml"
            
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load source config: {e}")
            return {'news_sources': {}}
    
    def _initialize_apis(self):
        """Initialize API clients"""
        from .news_apis import NewsAPIOrg, GuardianAPI
        
        # NewsAPI.org
        newsapi_key = settings.get_api_key('newsapi')
        if newsapi_key:
            self.apis['newsapi'] = NewsAPIOrg(newsapi_key)
            logger.info("NewsAPI client initialized")
        
        # Guardian API
        guardian_key = settings.get_api_key('guardian')
        if guardian_key:
            self.apis['guardian'] = GuardianAPI(guardian_key)
            logger.info("Guardian API client initialized")
        
        logger.info(f"✅ Initialized with APIs: {list(self.apis.keys())}")
    
    def collect_diverse_articles(self, query: str = "", days_back: int = 7,
                               strategy: str = "comprehensive") -> Dict[str, List[Article]]:
        """
        Collect articles from diverse sources using specified strategy
        
        Args:
            query: Search query
            days_back: Number of days to look back
            strategy: Collection strategy (comprehensive, mainstream_only, etc.)
            
        Returns:
            Dict mapping bias categories to article lists
        """
        logger.info(f"🔍 Collecting articles with strategy: {strategy}")
        logger.info(f"   Query: '{query}', Days back: {days_back}")
        
        results = {
            'left-leaning': [],
            'centrist': [], 
            'right-leaning': [],
            'libertarian': [],
            'international': []
        }
        
        # Get strategy config
        strategy_config = self.source_configs.get('collection_strategies', {}).get(
            strategy, {'include_tiers': ['major'], 'max_articles_per_source': 50}
        )
        
        included_tiers = strategy_config.get('include_tiers', ['major'])
        max_per_source = strategy_config.get('max_articles_per_source', 50)
        
        # Collect from each bias category
        for bias_category, sources in self.source_configs['news_sources'].items():
            logger.info(f"📰 Collecting {bias_category} articles...")
            
            # Filter sources by tier
            filtered_sources = [
                s for s in sources 
                if s.get('tier', 'major') in included_tiers
            ]
            
            logger.info(f"   Using {len(filtered_sources)} sources (tiers: {included_tiers})")
            
            category_articles = []
            
            # Collect from API sources
            api_sources = [s for s in filtered_sources if 'api_endpoint' in s]
            if api_sources:
                api_articles = self._collect_from_api_sources(
                    api_sources, query, days_back, max_per_source
                )
                category_articles.extend(api_articles)
            
            # Collect from RSS sources
            rss_sources = [s for s in filtered_sources if 'rss_feed' in s]
            if rss_sources:
                rss_articles = self._collect_from_rss_sources(
                    rss_sources, days_back, max_per_source
                )
                
                # Filter RSS articles by query if specified
                if query:
                    rss_articles = [
                        article for article in rss_articles
                        if query.lower() in article.title.lower() or 
                           (article.description and query.lower() in article.description.lower())
                    ]
                
                category_articles.extend(rss_articles)
            
            results[bias_category] = category_articles
            logger.info(f"   ✅ Collected {len(category_articles)} {bias_category} articles")
        
        # Log final summary
        total_articles = sum(len(articles) for articles in results.values())
        logger.info(f"📊 Total articles collected: {total_articles}")
        for category, articles in results.items():
            if articles:
                logger.info(f"   {category}: {len(articles)} articles")
        
        return results
    
    def _collect_from_api_sources(self, sources: List[Dict], query: str, 
                                days_back: int, max_per_source: int) -> List[Article]:
        """Collect articles from API sources"""
        articles = []
        
        # Group by API type
        newsapi_sources = [s for s in sources if 'newsapi.org' in s.get('api_endpoint', '')]
        guardian_sources = [s for s in sources if 'guardianapis.com' in s.get('api_endpoint', '')]
        
        # Collect from NewsAPI
        if newsapi_sources and 'newsapi' in self.apis:
            # Only get sources that have source_id
            valid_newsapi_sources = [s for s in newsapi_sources if 'source_id' in s]
            if valid_newsapi_sources:
                source_ids = [s['source_id'] for s in valid_newsapi_sources]
                logger.info(f"   📡 Fetching from NewsAPI: {source_ids}")
                
                try:
                    api_articles = self.apis['newsapi'].fetch_articles(
                        query=query,
                        sources=source_ids,
                        days_back=days_back
                    )
                    
                    # Add bias labels based on source
                    for article in api_articles:
                        for source_config in valid_newsapi_sources:
                            if (source_config.get('source_id', '').lower() in article.source.lower() or 
                                source_config['name'].lower() in article.source.lower()):
                                article.bias_label = source_config['bias_score']
                                break
                    
                    articles.extend(api_articles[:max_per_source])
                    logger.info(f"   ✅ Got {len(api_articles)} articles from NewsAPI")
                    
                except Exception as e:
                    logger.error(f"   ❌ NewsAPI collection failed: {e}")
            else:
                logger.info(f"   ⏭️  No valid NewsAPI sources found")
        
        # Collect from Guardian
        if guardian_sources and 'guardian' in self.apis:
            logger.info(f"   📡 Fetching from Guardian API")
            
            try:
                guardian_articles = self.apis['guardian'].fetch_articles(
                    query=query,
                    days_back=days_back
                )
                
                # Guardian is left-leaning
                for article in guardian_articles:
                    article.bias_label = 0
                
                articles.extend(guardian_articles[:max_per_source])
                logger.info(f"   ✅ Got {len(guardian_articles)} articles from Guardian")
                
            except Exception as e:
                logger.error(f"   ❌ Guardian API collection failed: {e}")
        
        return articles
    
    def _collect_from_rss_sources(self, sources: List[Dict], days_back: int, 
                                max_per_source: int) -> List[Article]:
        """Collect articles from RSS sources"""
        articles = []
        
        for source_config in sources:
            try:
                rss_url = source_config['rss_feed']
                source_name = source_config['name']
                bias_score = source_config['bias_score']
                
                source_articles = self.rss_collector.fetch_rss_articles(
                    rss_url, source_name, bias_score, days_back
                )
                
                # Limit articles per source
                articles.extend(source_articles[:max_per_source])
                
            except Exception as e:
                logger.error(f"   ❌ RSS collection failed for {source_config.get('name', 'unknown')}: {e}")
                continue
        
        return articles
    
    def get_available_sources(self, strategy: str = "comprehensive") -> Dict:
        """Get information about available sources for a strategy"""
        strategy_config = self.source_configs.get('collection_strategies', {}).get(
            strategy, {'include_tiers': ['major']}
        )
        
        included_tiers = strategy_config.get('include_tiers', ['major'])
        
        source_info = {
            'strategy': strategy,
            'included_tiers': included_tiers,
            'sources_by_category': {}
        }
        
        for bias_category, sources in self.source_configs['news_sources'].items():
            filtered_sources = [
                {
                    'name': s['name'],
                    'tier': s.get('tier', 'major'),
                    'type': 'API' if 'api_endpoint' in s else 'RSS',
                    'bias_score': s['bias_score']
                }
                for s in sources 
                if s.get('tier', 'major') in included_tiers
            ]
            
            source_info['sources_by_category'][bias_category] = filtered_sources
        
        return source_info
    
    def test_source_availability(self) -> Dict:
        """Test which sources are currently accessible"""
        availability = {
            'api_sources': {},
            'rss_sources': {},
            'summary': {
                'total_sources': 0,
                'available_sources': 0,
                'api_available': len(self.apis),
                'rss_tested': 0
            }
        }
        
        # Test API availability
        for api_name, api_client in self.apis.items():
            try:
                # Simple test call
                test_result = api_client.fetch_articles("test", [], days_back=1)
                availability['api_sources'][api_name] = {
                    'status': 'available',
                    'test_articles': len(test_result)
                }
            except Exception as e:
                availability['api_sources'][api_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Test a few RSS sources
        test_rss_sources = []
        for sources in self.source_configs['news_sources'].values():
            rss_sources = [s for s in sources if 'rss_feed' in s]
            test_rss_sources.extend(rss_sources[:2])  # Test 2 per category
        
        for source in test_rss_sources[:10]:  # Test max 10 RSS sources
            try:
                feed = feedparser.parse(source['rss_feed'])
                availability['rss_sources'][source['name']] = {
                    'status': 'available',
                    'entries': len(feed.entries)
                }
                availability['summary']['available_sources'] += 1
            except Exception as e:
                availability['rss_sources'][source['name']] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            availability['summary']['rss_tested'] += 1
            availability['summary']['total_sources'] += 1
        
        return availability

# Example usage and testing
if __name__ == "__main__":
    collector = EnhancedNewsCollector()
    
    # Test source availability
    print(" Testing source availability...")
    availability = collector.test_source_availability()
    print(f"API sources available: {len(availability['api_sources'])}")
    print(f"RSS sources tested: {availability['summary']['rss_tested']}")
    
    # Get available sources
    print("\n Available sources (comprehensive strategy):")
    sources = collector.get_available_sources("comprehensive")
    for category, source_list in sources['sources_by_category'].items():
        if source_list:
            print(f"\n{category.upper()}: {len(source_list)} sources")
            for source in source_list[:3]:  # Show first 3
                print(f"  • {source['name']} ({source['type']}, {source['tier']})")
    
    # Test collection
    print("\n🔍 Testing article collection...")
    articles = collector.collect_diverse_articles(query="election", days_back=3)
    
    total = sum(len(arts) for arts in articles.values())
    print(f"✅ Collected {total} total articles")
    for bias, arts in articles.items():
        if arts:
            print(f"  {bias}: {len(arts)} articles")