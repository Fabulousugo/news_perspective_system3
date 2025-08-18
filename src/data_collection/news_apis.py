# src/data_collection/news_apis.py - FIXED VERSION

import requests
import time
import logging
import os
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Article:
    """Standardized article data structure"""
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    author: Optional[str] = None
    description: Optional[str] = None
    bias_label: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'source': self.source,
            'published_at': self.published_at.isoformat(),
            'author': self.author,
            'description': self.description,
            'bias_label': self.bias_label
        }

class NewsAPIBase(ABC):
    """Abstract base class for news API integrations"""
    
    def __init__(self, api_key: str, rate_limit_delay: float = 1.0):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict) -> Dict:
        """Make HTTP request with error handling and retries"""
        self._rate_limit()
        
        MAX_RETRIES = 3
        API_TIMEOUT = 30
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Making request to {url} with params: {params}")
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=API_TIMEOUT
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
    @abstractmethod
    def fetch_articles(self, query: str, sources: List[str], 
                      days_back: int = 7) -> List[Article]:
        """Fetch articles from the API"""
        pass

class NewsAPIOrg(NewsAPIBase):
    """NewsAPI.org integration for mainstream sources"""
    
    BASE_URL = "https://newsapi.org/v2/"
    
    def fetch_articles(self, query: str = "", sources: List[str] = None, 
                      days_back: int = 7) -> List[Article]:
        """
        Fetch articles from NewsAPI.org
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Build parameters
        params = {
            'apiKey': self.api_key,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': 100
        }
        
        if query:
            params['q'] = query
            
        if sources:
            params['sources'] = ','.join(sources)
            endpoint = 'everything'
        else:
            endpoint = 'top-headlines'
            params['country'] = 'us'
        
        try:
            logger.info(f"Fetching from NewsAPI endpoint: {endpoint}")
            logger.info(f"Sources: {sources}")
            data = self._make_request(f"{self.BASE_URL}{endpoint}", params)
            return self._parse_newsapi_response(data)
            
        except Exception as e:
            logger.error(f"Failed to fetch from NewsAPI: {e}")
            return []
    
    def _parse_newsapi_response(self, data: Dict) -> List[Article]:
        """Parse NewsAPI.org response into Article objects"""
        articles = []
        
        if data.get('status') != 'ok':
            logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return articles
            
        for item in data.get('articles', []):
            try:
                # Skip articles without content
                if not item.get('content') or item['content'] == '[Removed]':
                    continue
                    
                # Parse publication date
                pub_date_str = item['publishedAt']
                if pub_date_str.endswith('Z'):
                    pub_date_str = pub_date_str[:-1] + '+00:00'
                pub_date = datetime.fromisoformat(pub_date_str)
                
                article = Article(
                    title=item['title'] or 'No title',
                    content=item['content'] or item.get('description', ''),
                    url=item['url'],
                    source=item['source']['name'],
                    published_at=pub_date,
                    author=item.get('author'),
                    description=item.get('description')
                )
                articles.append(article)
                
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
                continue
                
        logger.info(f"Parsed {len(articles)} articles from NewsAPI")
        return articles

class GuardianAPI(NewsAPIBase):
    """The Guardian API integration"""
    
    BASE_URL = "https://content.guardianapis.com/"
    
    def fetch_articles(self, query: str = "", sections: List[str] = None,
                      days_back: int = 7) -> List[Article]:
        """Fetch articles from The Guardian"""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'api-key': self.api_key,
            'from-date': from_date,
            'order-by': 'newest',
            'page-size': 50,
            'show-fields': 'bodyText,byline,publication'
        }
        
        if query:
            params['q'] = query
            
        if sections:
            params['section'] = '|'.join(sections)
        
        try:
            logger.info("Fetching from Guardian API")
            data = self._make_request(f"{self.BASE_URL}search", params)
            return self._parse_guardian_response(data)
            
        except Exception as e:
            logger.error(f"Failed to fetch from Guardian: {e}")
            return []
    
    def _parse_guardian_response(self, data: Dict) -> List[Article]:
        """Parse Guardian API response"""
        articles = []
        
        response = data.get('response', {})
        if response.get('status') != 'ok':
            logger.error("Guardian API error")
            return articles
            
        for item in response.get('results', []):
            try:
                fields = item.get('fields', {})
                content = fields.get('bodyText', '')
                
                if len(content) < 100:  # MIN_ARTICLE_LENGTH
                    continue
                
                pub_date_str = item['webPublicationDate']
                if pub_date_str.endswith('Z'):
                    pub_date_str = pub_date_str[:-1] + '+00:00'
                pub_date = datetime.fromisoformat(pub_date_str)
                
                article = Article(
                    title=item['webTitle'],
                    content=content,
                    url=item['webUrl'],
                    source='The Guardian',
                    published_at=pub_date,
                    author=fields.get('byline'),
                    bias_label=0  # Left-leaning
                )
                articles.append(article)
                
            except Exception as e:
                logger.warning(f"Failed to parse Guardian article: {e}")
                continue
                
        logger.info(f"Parsed {len(articles)} articles from Guardian")
        return articles

class NewsCollector:
    """Main news collection orchestrator - FIXED VERSION"""
    
    def __init__(self):
        self.apis = {}
        self.source_configs = self._load_source_configs()
        self._initialize_apis()
    
    def _load_source_configs(self) -> Dict:
        """Load source configurations with fallback"""
        try:
            # Try to load from YAML file
            import yaml
            from pathlib import Path
            
            config_file = Path(__file__).parent.parent.parent / "config" / "news_sources.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load YAML config: {e}")
        
        # Fallback to hardcoded configuration
        return {
            'news_sources': {
                'left_leaning': [
                    {'name': 'CNN', 'source_id': 'cnn', 'bias_score': 0},
                    {'name': 'The Guardian', 'source_id': 'the-guardian-uk', 'bias_score': 0},
                    {'name': 'MSNBC', 'source_id': 'msnbc', 'bias_score': 0},
                ],
                'centrist': [
                    {'name': 'Reuters', 'source_id': 'reuters', 'bias_score': 1},
                    {'name': 'Associated Press', 'source_id': 'associated-press', 'bias_score': 1},
                    {'name': 'BBC News', 'source_id': 'bbc-news', 'bias_score': 1},
                ],
                'right_leaning': [
                    {'name': 'Fox News', 'source_id': 'fox-news', 'bias_score': 2},
                    {'name': 'New York Post', 'source_id': 'new-york-post', 'bias_score': 2},
                    {'name': 'The Wall Street Journal', 'source_id': 'the-wall-street-journal', 'bias_score': 2},
                ]
            }
        }
    
    def _initialize_apis(self):
        """Initialize API clients with keys"""
        # Try different ways to get API keys
        newsapi_key = (
            os.getenv('NEWSAPI_API_KEY') or 
            os.getenv('NEWSAPI_KEY') or 
            os.getenv('NEWS_API_KEY')
        )
        
        guardian_key = (
            os.getenv('GUARDIAN_API_KEY') or 
            os.getenv('GUARDIAN_KEY')
        )
        
        logger.info(f"NewsAPI key found: {bool(newsapi_key)}")
        logger.info(f"Guardian key found: {bool(guardian_key)}")
        
        if newsapi_key:
            self.apis['newsapi'] = NewsAPIOrg(newsapi_key)
            logger.info("NewsAPI client initialized")
        else:
            logger.warning("No NewsAPI key found")
            
        if guardian_key:
            self.apis['guardian'] = GuardianAPI(guardian_key)
            logger.info("Guardian API client initialized")
        else:
            logger.warning("No Guardian API key found")
        
        if not self.apis:
            logger.error("No API clients initialized! Check your API keys.")
    
    def collect_diverse_articles(self, query: str = "", 
                               days_back: int = 7) -> Dict[str, List[Article]]:
        """
        Collect articles from diverse political perspectives
        """
        results = {
            'left_leaning': [],
            'centrist': [],
            'right_leaning': []
        }
        
        if not self.apis:
            logger.error("No APIs available for collection")
            return results
        
        # Collect from each bias category
        for bias_category, sources in self.source_configs['news_sources'].items():
            logger.info(f"Collecting {bias_category} articles...")
            
            if 'newsapi' in self.apis:
                # Extract source IDs for NewsAPI
                # newsapi_sources = [s['source_id'] for s in sources]
                newsapi_sources = [s['source_id'] for s in sources if 'source_id' in s]
                
                try:
                    articles = self.apis['newsapi'].fetch_articles(
                        query=query,
                        sources=newsapi_sources,
                        days_back=days_back
                    )
                    
                    # Add bias labels
                    for article in articles:
                        if sources:  # Make sure sources list is not empty
                            article.bias_label = sources[0]['bias_score']
                    
                    results[bias_category].extend(articles)
                    logger.info(f"Added {len(articles)} {bias_category} articles")
                    
                except Exception as e:
                    logger.error(f"Failed to collect {bias_category} articles: {e}")
            
            # Handle Guardian separately if present and it's in left_leaning
            if (bias_category == 'left_leaning' and 
                'guardian' in self.apis and 
                any(s['name'] == 'The Guardian' for s in sources)):
                
                try:
                    guardian_articles = self.apis['guardian'].fetch_articles(
                        query=query,
                        days_back=days_back
                    )
                    results[bias_category].extend(guardian_articles)
                    logger.info(f"Added {len(guardian_articles)} Guardian articles")
                    
                except Exception as e:
                    logger.error(f"Failed to collect Guardian articles: {e}")
        
        # Log collection summary
        total_articles = sum(len(articles) for articles in results.values())
        logger.info(f"Collected {total_articles} total articles")
        for category, articles in results.items():
            logger.info(f"  {category}: {len(articles)} articles")
            
        return results
    
    def get_articles_by_topic(self, topic: str, days_back: int = 7) -> List[Article]:
        """Get articles about a specific topic from all sources"""
        all_articles = []
        diverse_results = self.collect_diverse_articles(query=topic, days_back=days_back)
        
        for articles in diverse_results.values():
            all_articles.extend(articles)
            
        # Sort by publication date
        all_articles.sort(key=lambda x: x.published_at, reverse=True)
        return all_articles

# Quick test function
def test_collector():
    """Test the news collector"""
    print("Testing NewsCollector...")
    
    collector = NewsCollector()
    
    if not collector.apis:
        print("❌ No APIs initialized")
        return False
    
    print(f"✅ APIs initialized: {list(collector.apis.keys())}")
    
    # Test with simple query
    articles = collector.collect_diverse_articles(query="", days_back=1)
    
    total = sum(len(arts) for arts in articles.values())
    print(f"Collected {total} articles:")
    
    for bias, arts in articles.items():
        print(f"  {bias}: {len(arts)} articles")
        if arts:
            print(f"    Sample: {arts[0].title}")
    
    return total > 0

if __name__ == "__main__":
    test_collector()