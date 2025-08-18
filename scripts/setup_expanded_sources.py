# scripts/setup_expanded_sources.py - Setup script for enhanced news sources

import shutil
from pathlib import Path
import yaml

def setup_expanded_sources():
    """Set up the expanded news sources configuration"""
    
    print("🚀 Setting up Enhanced News Source Collection")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"
    
    # Ensure config directory exists
    config_dir.mkdir(exist_ok=True)
    
    # Create expanded sources config
    expanded_config = {
        'news_sources': {
            'left_leaning': [
                {'name': 'CNN', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'cnn', 'bias_score': 0, 'tier': 'major'},
                {'name': 'The Guardian', 'api_endpoint': 'https://content.guardianapis.com/search', 'source_id': 'the-guardian-uk', 'bias_score': 0, 'tier': 'major'},
                {'name': 'MSNBC', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'msnbc', 'bias_score': 0, 'tier': 'major'},
                {'name': 'NPR', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'npr', 'bias_score': 0, 'tier': 'major'},
                {'name': 'The Washington Post', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'the-washington-post', 'bias_score': 0, 'tier': 'major'},
                {'name': 'The New York Times', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'the-new-york-times', 'bias_score': 0, 'tier': 'major'},
                {'name': 'HuffPost', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'the-huffington-post', 'bias_score': 0, 'tier': 'secondary'},
                {'name': 'Politico', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'politico', 'bias_score': 0, 'tier': 'secondary'},
                {'name': 'The Hill', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'the-hill', 'bias_score': 0, 'tier': 'secondary'},
                {'name': 'Salon', 'rss_feed': 'https://www.salon.com/feed/', 'bias_score': 0, 'tier': 'alternative'},
                {'name': 'Mother Jones', 'rss_feed': 'https://www.motherjones.com/feed/', 'bias_score': 0, 'tier': 'alternative'},
                {'name': 'The Nation', 'rss_feed': 'https://www.thenation.com/feed/', 'bias_score': 0, 'tier': 'alternative'},
                {'name': 'The Independent (UK)', 'rss_feed': 'https://www.independent.co.uk/rss', 'bias_score': 0, 'tier': 'international'}
            ],
            'centrist': [
                {'name': 'Reuters', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'reuters', 'bias_score': 1, 'tier': 'major'},
                {'name': 'Associated Press', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'associated-press', 'bias_score': 1, 'tier': 'major'},
                {'name': 'BBC News', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'bbc-news', 'bias_score': 1, 'tier': 'major'},
                {'name': 'USA Today', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'usa-today', 'bias_score': 1, 'tier': 'major'},
                {'name': 'CBS News', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'cbs-news', 'bias_score': 1, 'tier': 'major'},
                {'name': 'ABC News', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'abc-news', 'bias_score': 1, 'tier': 'major'},
                {'name': 'Al Jazeera English', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'al-jazeera-english', 'bias_score': 1, 'tier': 'international'},
                {'name': 'Bloomberg', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'bloomberg', 'bias_score': 1, 'tier': 'business'},
                {'name': 'Financial Times', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'financial-times', 'bias_score': 1, 'tier': 'business'},
                {'name': 'Deutsche Welle', 'rss_feed': 'https://rss.dw.com/xml/rss-en-all', 'bias_score': 1, 'tier': 'international'},
                {'name': 'France 24', 'rss_feed': 'https://www.france24.com/en/rss', 'bias_score': 1, 'tier': 'international'}
            ],
            'right_leaning': [
                {'name': 'Fox News', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'fox-news', 'bias_score': 2, 'tier': 'major'},
                {'name': 'New York Post', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'new-york-post', 'bias_score': 2, 'tier': 'major'},
                {'name': 'Wall Street Journal', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'the-wall-street-journal', 'bias_score': 2, 'tier': 'major'},
                {'name': 'Washington Examiner', 'api_endpoint': 'https://newsapi.org/v2/everything', 'source_id': 'washington-examiner', 'bias_score': 2, 'tier': 'major'},
                {'name': 'The Daily Wire', 'rss_feed': 'https://www.dailywire.com/feeds/rss.xml', 'bias_score': 2, 'tier': 'alternative'},
                {'name': 'Breitbart News', 'rss_feed': 'https://feeds.feedburner.com/breitbart', 'bias_score': 2, 'tier': 'alternative'},
                {'name': 'The Federalist', 'rss_feed': 'https://thefederalist.com/feed/', 'bias_score': 2, 'tier': 'alternative'},
                {'name': 'National Review', 'rss_feed': 'https://www.nationalreview.com/feed/', 'bias_score': 2, 'tier': 'conservative'},
                {'name': 'American Conservative', 'rss_feed': 'https://www.theamericanconservative.com/feed/', 'bias_score': 2, 'tier': 'conservative'},
                {'name': 'Washington Times', 'rss_feed': 'https://www.washingtontimes.com/rss/headlines/', 'bias_score': 2, 'tier': 'secondary'},
                {'name': 'The Telegraph (UK)', 'rss_feed': 'https://www.telegraph.co.uk/rss.xml', 'bias_score': 2, 'tier': 'international'}
            ],
            'libertarian': [
                {'name': 'Reason', 'rss_feed': 'https://reason.com/feed/', 'bias_score': 3, 'tier': 'alternative'},
                {'name': 'Cato Institute', 'rss_feed': 'https://www.cato.org/rss/commentary', 'bias_score': 3, 'tier': 'think_tank'}
            ]
        },
        'api_configs': {
            'newsapi': {
                'base_url': 'https://newsapi.org/v2/',
                'rate_limit': 1000,
                'requires_key': True
            },
            'guardian': {
                'base_url': 'https://content.guardianapis.com/',
                'rate_limit': 5000,
                'requires_key': True
            },
            'rss': {
                'rate_limit': 100,
                'requires_key': False,
                'cache_duration': 3600
            }
        },
        'source_tiers': {
            'major': 'Primary mainstream sources',
            'secondary': 'Secondary mainstream sources',
            'alternative': 'Alternative/opinion sources', 
            'business': 'Business/financial focus',
            'international': 'International sources',
            'think_tank': 'Think tanks and policy institutes',
            'conservative': 'Conservative publications'
        },
        'collection_strategies': {
            'comprehensive': {
                'include_tiers': ['major', 'secondary', 'business', 'international'],
                'max_articles_per_source': 50
            },
            'mainstream_only': {
                'include_tiers': ['major'],
                'max_articles_per_source': 100
            },
            'alternative_focus': {
                'include_tiers': ['alternative', 'conservative', 'think_tank'],
                'max_articles_per_source': 30
            },
            'international': {
                'include_tiers': ['international', 'major'],
                'max_articles_per_source': 25
            }
        }
    }
    
    # Write expanded config
    expanded_config_path = config_dir / "expanded_news_sources.yaml"
    with open(expanded_config_path, 'w') as f:
        yaml.dump(expanded_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created expanded sources config: {expanded_config_path}")
    
    # Count sources
    total_sources = 0
    api_sources = 0
    rss_sources = 0
    
    for bias_category, sources in expanded_config['news_sources'].items():
        total_sources += len(sources)
        for source in sources:
            if 'api_endpoint' in source:
                api_sources += 1
            elif 'rss_feed' in source:
                rss_sources += 1
    
    print(f"\n📊 Source Summary:")
    print(f"   Total sources: {total_sources}")
    print(f"   API sources: {api_sources}")
    print(f"   RSS sources: {rss_sources}")
    
    # Show breakdown by bias
    print(f"\n🏛️  Bias Category Breakdown:")
    for bias_category, sources in expanded_config['news_sources'].items():
        print(f"   {bias_category}: {len(sources)} sources")
    
    # Show tier breakdown
    print(f"\n📋 Tier Breakdown:")
    tier_counts = {}
    for sources in expanded_config['news_sources'].values():
        for source in sources:
            tier = source.get('tier', 'unknown')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    for tier, count in sorted(tier_counts.items()):
        print(f"   {tier}: {count} sources")
    
    print(f"\n🚀 Setup Complete!")
    print(f"\nNext steps:")
    print(f"1. Test source connectivity:")
    print(f"   python scripts/improved_news_browser.py sources")
    print(f"2. Test collection with different strategies:")
    print(f"   python scripts/improved_news_browser.py test-collection --strategy comprehensive")
    print(f"3. Browse news with enhanced diversity:")
    print(f"   python scripts/improved_news_browser.py browse --strategy comprehensive --days 14")
    
    return True

if __name__ == "__main__":
    setup_expanded_sources()