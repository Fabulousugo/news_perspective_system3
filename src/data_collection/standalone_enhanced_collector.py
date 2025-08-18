# src/data_collection/standalone_enhanced_collector.py - Completely standalone

import requests
import feedparser
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re
import os,sys


# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ..data_collection.news_apis import NewsAPIOrg, GuardianAPI, Article
from config.settings import settings

logger = logging.getLogger(__name__)

class StandaloneEnhancedCollector:
    """
    Completely standalone news collector with enhanced sources
    No inheritance issues - builds everything from scratch
    """
    
    def __init__(self):
        # Initialize API clients
        self.apis = {}
        self._initialize_apis()
        
        # Define all sources in one place - no external config files
        self.source_config = {
            'left_leaning': {
                'api_sources': [
                    {'name': 'CNN', 'source_id': 'cnn', 'bias': 0},
                    {'name': 'MSNBC', 'source_id': 'msnbc', 'bias': 0},
                    {'name': 'NPR', 'source_id': 'npr', 'bias': 0},
                    {'name': 'The Washington Post', 'source_id': 'the-washington-post', 'bias': 0},
                    {'name': 'HuffPost', 'source_id': 'the-huffington-post', 'bias': 0},
                ],
                'rss_sources': [
                    {'name': 'Salon', 'url': 'https://www.salon.com/feed/', 'bias': 0},
                    {'name': 'Mother Jones', 'url': 'https://www.motherjones.com/feed/', 'bias': 0},
                    {'name': 'The Nation', 'url': 'https://www.thenation.com/feed/', 'bias': 0},
                ]
            },
            'centrist': {
                'api_sources': [
                    {'name': 'Reuters', 'source_id': 'reuters', 'bias': 1},
                    {'name': 'Associated Press', 'source_id': 'associated-press', 'bias': 1},
                    {'name': 'BBC News', 'source_id': 'bbc-news', 'bias': 1},
                    {'name': 'USA Today', 'source_id': 'usa-today', 'bias': 1},
                    {'name': 'Bloomberg', 'source_id': 'bloomberg', 'bias': 1},
                ],
                'rss_sources': [
                    {'name': 'Al Jazeera', 'url': 'https://www.aljazeera.com/xml/rss/all.xml', 'bias': 1},
                    {'name': 'PBS NewsHour', 'url': 'https://www.pbs.org/newshour/feeds/rss/headlines', 'bias': 1},
                ]
            },
            'right_leaning': {
                'api_sources': [
                    {'name': 'Fox News', 'source_id': 'fox-news', 'bias': 2},
                    {'name': 'New York Post', 'source_id': 'new-york-post', 'bias': 2},
                    {'name': 'Wall Street Journal', 'source_id': 'the-wall-street-journal', 'bias': 2},
                    {'name': 'Washington Examiner', 'source_id': 'washington-examiner', 'bias': 2},
                ],
                'rss_sources': [
                    {'name': 'The Daily Wire', 'url': 'https://www.dailywire.com/feeds/rss.xml', 'bias': 2},
                    {'name': 'Breitbart', 'url': 'https://feeds.feedburner.com/breitbart', 'bias': 2},
                    {'name': 'National Review', 'url': 'https://www.nationalreview.com/feed/', 'bias': 2},
                    {'name': 'The Federalist', 'url': 'https://thefederalist.com/feed/', 'bias': 2},
                    {'name': 'Washington Times', 'url': 'https://www.washingtontimes.com/rss/headlines/', 'bias': 2},
                ]
            }
        }
        
        # Count sources
        total_api = sum(len(cat['api_sources']) for cat in self.source_config.values())
        total_rss = sum(len(cat['rss_sources']) for cat in self.source_config.values())
        
        logger.info(f"✅ Standalone collector initialized:")
        logger.info(f"   API sources: {total_api}")
        logger.info(f"   RSS sources: {total_rss}")
        logger.info(f"   Total sources: {total_api + total_rss}")
    
    def _initialize_apis(self):
        """Initialize API clients safely"""
        try:
            # NewsAPI
            newsapi_key = os.getenv('NEWSAPI_API_KEY')
            if newsapi_key:
                self.apis['newsapi'] = NewsAPIOrg(newsapi_key)
                logger.info("✅ NewsAPI client initialized")
            else:
                logger.warning("⚠️  No NewsAPI key found")
            
            # Guardian API
            guardian_key = os.getenv('GUARDIAN_API_KEY')
            if guardian_key:
                self.apis['guardian'] = GuardianAPI(guardian_key)
                logger.info("✅ Guardian API client initialized")
            else:
                logger.warning("⚠️  No Guardian key found")
                
        except Exception as e:
            logger.error(f"❌ API initialization error: {e}")
    
    def collect_diverse_articles(self, query: str = "", days_back: int = 7) -> Dict[str, List[Article]]:
        """
        Collect articles from all configured sources
        """
        logger.info(f"🔍 Collecting articles: query='{query}', days_back={days_back}")
        
        results = {
            'left_leaning': [],
            'centrist': [],
            'right_leaning': []
        }
        
        for bias_category, category_config in self.source_config.items():
            logger.info(f"📰 Collecting {bias_category} articles...")
            
            category_articles = []
            
            # Collect from API sources
            api_articles = self._collect_api_articles(
                category_config['api_sources'], 
                query, 
                days_back
            )
            category_articles.extend(api_articles)
            
            # Collect from RSS sources
            rss_articles = self._collect_rss_articles(
                category_config['rss_sources'], 
                query, 
                days_back
            )
            category_articles.extend(rss_articles)
            
            results[bias_category] = category_articles
            logger.info(f"   ✅ {bias_category}: {len(category_articles)} articles")
        
        total = sum(len(arts) for arts in results.values())
        logger.info(f"📊 Total collected: {total} articles")
        
        return results
    
    def _collect_api_articles(self, api_sources: List[Dict], query: str, days_back: int) -> List[Article]:
        """Collect from API sources"""
        articles = []
        
        if not api_sources or 'newsapi' not in self.apis:
            return articles
        
        try:
            # Get source IDs
            source_ids = [source['source_id'] for source in api_sources]
            logger.info(f"   📡 API sources: {source_ids}")
            
            # Fetch from NewsAPI
            api_articles = self.apis['newsapi'].fetch_articles(
                query=query,
                sources=source_ids,
                days_back=days_back
            )
            
            # Add bias labels
            for article in api_articles:
                for source_config in api_sources:
                    if (source_config['source_id'].lower() in article.source.lower() or
                        source_config['name'].lower() in article.source.lower()):
                        article.bias_label = source_config['bias']
                        break
            
            articles.extend(api_articles)
            logger.info(f"   📡 API: {len(api_articles)} articles")
            
        except Exception as e:
            logger.error(f"   ❌ API collection failed: {e}")
        
        # Also try Guardian for left-leaning if available
        if ('guardian' in self.apis and 
            any(source['bias'] == 0 for source in api_sources)):
            try:
                guardian_articles = self.apis['guardian'].fetch_articles(
                    query=query,
                    days_back=days_back
                )
                
                # Guardian is left-leaning
                for article in guardian_articles:
                    article.bias_label = 0
                    article.source = "The Guardian"
                
                articles.extend(guardian_articles[:20])  # Limit Guardian articles
                logger.info(f"   📡 Guardian: {len(guardian_articles)} articles")
                
            except Exception as e:
                logger.error(f"   ❌ Guardian collection failed: {e}")
        
        return articles
    
    def _collect_rss_articles(self, rss_sources: List[Dict], query: str, days_back: int) -> List[Article]:
        """Collect from RSS sources"""
        articles = []
        
        for source_config in rss_sources:
            try:
                source_articles = self._fetch_single_rss(
                    source_config['url'],
                    source_config['name'],
                    source_config['bias'],
                    days_back,
                    query
                )
                articles.extend(source_articles)
                logger.info(f"   📰 {source_config['name']}: {len(source_articles)} articles")
                
            except Exception as e:
                logger.warning(f"   ⚠️  {source_config['name']} failed: {e}")
                continue
        
        return articles
    
    def _fetch_single_rss(self, rss_url: str, source_name: str, bias_score: int, 
                         days_back: int, query: str = "") -> List[Article]:
        """Fetch articles from a single RSS feed"""
        
        # Rate limiting
        time.sleep(1)
        
        try:
            # Fetch RSS with timeout
            response = requests.get(rss_url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            # Parse feed
            feed = feedparser.parse(response.content)
            
            articles = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for entry in feed.entries[:15]:  # Limit entries per source
                try:
                    # Parse date
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    else:
                        pub_date = datetime.now()
                    
                    # Skip old articles
                    if pub_date < cutoff_date:
                        continue
                    
                    # Filter by query
                    if query:
                        title_text = entry.title.lower()
                        summary_text = getattr(entry, 'summary', '').lower()
                        if query.lower() not in title_text and query.lower() not in summary_text:
                            continue
                    
                    # Extract content
                    content = ""
                    if hasattr(entry, 'content') and entry.content:
                        content = entry.content[0].value
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    
                    # Clean HTML
                    content = self._clean_html(content)
                    description = self._clean_html(getattr(entry, 'summary', ''))
                    
                    # Skip very short content
                    if len(content) < 30 and len(description) < 30:
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
                    logger.warning(f"Failed to parse entry from {source_name}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch RSS from {source_name}: {e}")
            return []
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and entities"""
        if not text:
            return ""
        
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', text)
        
        # Replace common HTML entities
        entities = {
            '&amp;': '&', '&lt;': '<', '&gt;': '>', 
            '&quot;': '"', '&#39;': "'", '&nbsp;': ' ',
            '&mdash;': '—', '&ndash;': '–', '&hellip;': '...'
        }
        
        for entity, replacement in entities.items():
            clean = clean.replace(entity, replacement)
        
        # Clean whitespace
        clean = ' '.join(clean.split())
        
        return clean.strip()
    
    def get_source_summary(self) -> Dict:
        """Get summary of all configured sources"""
        summary = {
            'api_sources': {},
            'rss_sources': {},
            'totals': {'api': 0, 'rss': 0, 'total': 0}
        }
        
        for bias_category, category_config in self.source_config.items():
            # API sources
            api_names = [s['name'] for s in category_config['api_sources']]
            summary['api_sources'][bias_category] = api_names
            summary['totals']['api'] += len(api_names)
            
            # RSS sources
            rss_names = [s['name'] for s in category_config['rss_sources']]
            summary['rss_sources'][bias_category] = rss_names
            summary['totals']['rss'] += len(rss_names)
        
        summary['totals']['total'] = summary['totals']['api'] + summary['totals']['rss']
        
        return summary
    
    def test_connectivity(self) -> Dict:
        """Test API and RSS connectivity"""
        results = {
            'api_status': {},
            'rss_status': {},
            'summary': {'api_available': 0, 'rss_available': 0}
        }
        
        # Test APIs
        for api_name, api_client in self.apis.items():
            try:
                test_articles = api_client.fetch_articles("test", [], days_back=1)
                results['api_status'][api_name] = f"✅ Working ({len(test_articles)} test articles)"
                results['summary']['api_available'] += 1
            except Exception as e:
                results['api_status'][api_name] = f"❌ Error: {str(e)[:50]}..."
        
        # Test a few RSS feeds
        test_rss = [
            ('Salon', 'https://www.salon.com/feed/'),
            ('Daily Wire', 'https://www.dailywire.com/feeds/rss.xml'),
            ('Al Jazeera', 'https://www.aljazeera.com/xml/rss/all.xml')
        ]
        
        for name, url in test_rss:
            try:
                response = requests.get(url, timeout=10)
                feed = feedparser.parse(response.content)
                if len(feed.entries) > 0:
                    results['rss_status'][name] = f"✅ Working ({len(feed.entries)} entries)"
                    results['summary']['rss_available'] += 1
                else:
                    results['rss_status'][name] = "⚠️  Empty feed"
            except Exception as e:
                results['rss_status'][name] = f"❌ Error: {str(e)[:50]}..."
        
        return results

# Example usage
if __name__ == "__main__":
    collector = StandaloneEnhancedCollector()
    
    # Test connectivity
    print("🧪 Testing connectivity...")
    connectivity = collector.test_connectivity()
    
    print("📡 API Status:")
    for api, status in connectivity['api_status'].items():
        print(f"   {api}: {status}")
    
    print("📰 RSS Status (sample):")
    for source, status in connectivity['rss_status'].items():
        print(f"   {source}: {status}")
    
    # Test collection
    print("\n📊 Testing collection...")
    articles = collector.collect_diverse_articles("election", days_back=3)
    
    total = sum(len(arts) for arts in articles.values())
    print(f"✅ Total articles: {total}")
    
    for bias, arts in articles.items():
        if arts:
            sources = list(set(a.source for a in arts))
            print(f"   {bias}: {len(arts)} articles from {', '.join(sources)}")