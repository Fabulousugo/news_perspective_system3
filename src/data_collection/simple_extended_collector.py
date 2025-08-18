# src/data_collection/simple_extended_collector.py - Bulletproof extension

import requests
import feedparser
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re

from .news_apis import NewsCollector, Article
from config.settings import settings

logger = logging.getLogger(__name__)

class SimpleExtendedCollector(NewsCollector):
    """
    Simple extension of the working NewsCollector with additional RSS sources
    """
    
    def __init__(self):
        super().__init__()
        
        # Additional RSS sources to supplement API sources
        self.rss_sources = {
            'left_leaning': [
                {'name': 'Salon', 'url': 'https://www.salon.com/feed/', 'bias': 0},
                {'name': 'Mother Jones', 'url': 'https://www.motherjones.com/feed/', 'bias': 0},
                {'name': 'HuffPost', 'url': 'https://www.huffpost.com/section/front-page/feed', 'bias': 0},
            ],
            'centrist': [
                {'name': 'Al Jazeera', 'url': 'https://www.aljazeera.com/xml/rss/all.xml', 'bias': 1},
                {'name': 'PBS NewsHour', 'url': 'https://www.pbs.org/newshour/feeds/rss/headlines', 'bias': 1},
            ],
            'right_leaning': [
                {'name': 'The Daily Wire', 'url': 'https://www.dailywire.com/feeds/rss.xml', 'bias': 2},
                {'name': 'Breitbart', 'url': 'https://feeds.feedburner.com/breitbart', 'bias': 2},
                {'name': 'National Review', 'url': 'https://www.nationalreview.com/feed/', 'bias': 2},
                {'name': 'The Federalist', 'url': 'https://thefederalist.com/feed/', 'bias': 2},
                {'name': 'Washington Times', 'url': 'https://www.washingtontimes.com/rss/headlines/', 'bias': 2},
            ]
        }
        
        logger.info(f"✅ Extended collector initialized with {sum(len(sources) for sources in self.rss_sources.values())} additional RSS sources")
    
    def collect_diverse_articles(self, query: str = "", days_back: int = 7) -> Dict[str, List[Article]]:
        """
        Collect articles using original API method + additional RSS sources
        """
        logger.info(f"🔍 Collecting articles with extended sources...")
        
        # First, get articles using the working API method
        api_results = super().collect_diverse_articles(query, days_back)
        
        # Then, add RSS articles
        for bias_category, rss_source_list in self.rss_sources.items():
            logger.info(f"📰 Adding RSS sources for {bias_category}...")
            
            rss_articles = []
            for source_config in rss_source_list:
                try:
                    articles = self._fetch_rss_articles(
                        source_config['url'], 
                        source_config['name'], 
                        source_config['bias'], 
                        days_back,
                        query
                    )
                    rss_articles.extend(articles)
                    
                except Exception as e:
                    logger.warning(f"   ⚠️  Failed to fetch from {source_config['name']}: {e}")
                    continue
            
            # Add to results
            if bias_category in api_results:
                api_results[bias_category].extend(rss_articles)
            else:
                api_results[bias_category] = rss_articles
            
            logger.info(f"   ✅ Added {len(rss_articles)} RSS articles for {bias_category}")
        
        # Log final totals
        total_articles = sum(len(articles) for articles in api_results.values())
        logger.info(f"📊 Extended collection complete: {total_articles} total articles")
        for category, articles in api_results.items():
            if articles:
                sources = list(set(a.source for a in articles))
                logger.info(f"   {category}: {len(articles)} articles from {len(sources)} sources")
        
        return api_results
    
    def _fetch_rss_articles(self, rss_url: str, source_name: str, bias_score: int, 
                          days_back: int, query: str = "") -> List[Article]:
        """
        Fetch articles from a single RSS feed
        """
        # Rate limiting
        time.sleep(1)  # Simple rate limiting
        
        try:
            # Fetch and parse RSS
            response = requests.get(rss_url, timeout=10)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            if feed.bozo and feed.bozo_exception:
                logger.warning(f"RSS feed issues for {source_name}: {feed.bozo_exception}")
            
            articles = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for entry in feed.entries[:20]:  # Limit to first 20 entries
                try:
                    # Parse publication date
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    else:
                        pub_date = datetime.now()
                    
                    # Skip old articles
                    if pub_date < cutoff_date:
                        continue
                    
                    # Filter by query if specified
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
                    
                    # Skip if too short
                    if len(content) < 50 and len(description) < 50:
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
            
            logger.info(f"   📄 {source_name}: {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch RSS from {source_name}: {e}")
            return []
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        if not text:
            return ""
        
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', text)
        
        # Replace HTML entities
        replacements = {
            '&amp;': '&', '&lt;': '<', '&gt;': '>', 
            '&quot;': '"', '&#39;': "'", '&nbsp;': ' '
        }
        
        for entity, replacement in replacements.items():
            clean = clean.replace(entity, replacement)
        
        # Clean up whitespace
        clean = ' '.join(clean.split())
        
        return clean.strip()
    
    def get_source_summary(self) -> Dict:
        """Get summary of available sources"""
        api_summary = {
            'api_sources': list(self.apis.keys()),
            'rss_sources': {}
        }
        
        for bias_category, sources in self.rss_sources.items():
            api_summary['rss_sources'][bias_category] = [s['name'] for s in sources]
        
        # Count totals
        total_rss = sum(len(sources) for sources in self.rss_sources.values())
        
        api_summary['summary'] = {
            'total_api_sources': len(self.apis),
            'total_rss_sources': total_rss,
            'total_sources': len(self.apis) + total_rss
        }
        
        return api_summary

# Example usage and testing
if __name__ == "__main__":
    collector = SimpleExtendedCollector()
    
    # Show available sources
    summary = collector.get_source_summary()
    print("📊 Extended Source Summary:")
    print(f"   API sources: {summary['summary']['total_api_sources']}")
    print(f"   RSS sources: {summary['summary']['total_rss_sources']}")
    print(f"   Total sources: {summary['summary']['total_sources']}")
    
    print(f"\n📰 RSS Sources by bias:")
    for bias, sources in summary['rss_sources'].items():
        print(f"   {bias}: {', '.join(sources)}")
    
    # Test collection
    print(f"\n🧪 Testing collection...")
    articles = collector.collect_diverse_articles("election", days_back=3)
    
    total = sum(len(arts) for arts in articles.values())
    print(f"✅ Collected {total} total articles")
    
    for bias, arts in articles.items():
        if arts:
            sources = list(set(a.source for a in arts))
            print(f"   {bias}: {len(arts)} articles from sources: {', '.join(sources)}")