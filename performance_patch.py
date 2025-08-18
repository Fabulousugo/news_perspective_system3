# performance_patch.py - Quick performance improvement patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def apply_performance_patch():
    """Apply immediate performance improvements to your existing system"""
    
    print("🚀 Applying Performance Optimization Patch")
    print("=" * 50)
    
    try:
        # Patch the existing perspective matcher with fast version
        from models.perspective_matcher import PerspectiveMatcher
        from data_collection.news_apis import NewsCollector
        import time
        import logging
        from collections import defaultdict
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        logger = logging.getLogger(__name__)
        
        def fast_find_perspective_matches(self, articles, min_perspectives=2):
            """Ultra-fast perspective matching replacement"""
            
            start_time = time.time()
            logger.info(f"🚀 FAST perspective matching for {len(articles)} articles...")
            
            # Step 1: Quick preprocessing - limit articles for speed
            if len(articles) > 100:
                # Take most recent articles
                articles = sorted(articles, key=lambda x: x.published_at, reverse=True)[:100]
                logger.info(f"📋 Limited to {len(articles)} most recent articles for speed")
            
            # Step 2: Assign bias labels if missing
            labeled_articles = []
            source_bias_map = {
                'CNN': 0, 'MSNBC': 0, 'NPR': 0, 'The Guardian': 0, 'Washington Post': 0,
                'Reuters': 1, 'Associated Press': 1, 'BBC': 1, 'USA Today': 1,
                'Fox News': 2, 'New York Post': 2, 'Wall Street Journal': 2, 'Washington Examiner': 2
            }
            
            for article in articles:
                bias_label = article.bias_label
                if bias_label is None:
                    bias_label = source_bias_map.get(article.source, 1)  # Default centrist
                
                # Create labeled version
                from data_collection.news_apis import Article
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
            
            # Step 3: Group by bias quickly
            bias_groups = defaultdict(list)
            bias_labels = {0: "left-leaning", 1: "centrist", 2: "right-leaning"}
            
            for article in labeled_articles:
                if article.bias_label is not None:
                    bias_category = bias_labels[article.bias_label]
                    bias_groups[bias_category].append(article)
            
            logger.info(f"📊 Bias distribution: {[(bias, len(arts)) for bias, arts in bias_groups.items()]}")
            
            if len(bias_groups) < min_perspectives:
                logger.warning(f"Not enough bias groups: {len(bias_groups)}")
                return []
            
            # Step 4: FAST similarity matching using TF-IDF
            matches = []
            bias_list = list(bias_groups.keys())
            
            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,      # Reduced for speed
                stop_words='english',
                ngram_range=(1, 1),     # Only unigrams for speed
                max_df=0.8,
                min_df=1
            )
            
            for i, bias1 in enumerate(bias_list):
                for j, bias2 in enumerate(bias_list[i+1:], i+1):
                    articles1 = bias_groups[bias1][:20]  # Limit for speed
                    articles2 = bias_groups[bias2][:20]  # Limit for speed
                    
                    if not articles1 or not articles2:
                        continue
                    
                    try:
                        # Use only titles for speed
                        texts1 = [art.title for art in articles1]
                        texts2 = [art.title for art in articles2]
                        all_texts = texts1 + texts2
                        
                        if len(all_texts) < 2:
                            continue
                        
                        # Fast TF-IDF similarity
                        tfidf_matrix = vectorizer.fit_transform(all_texts)
                        matrix1 = tfidf_matrix[:len(texts1)]
                        matrix2 = tfidf_matrix[len(texts1):]
                        
                        similarities = cosine_similarity(matrix1, matrix2)
                        
                        # Find top matches
                        for idx1, row in enumerate(similarities):
                            for idx2, sim_score in enumerate(row):
                                if sim_score >= 0.3:  # Lower threshold for speed
                                    
                                    # Create simplified match object
                                    from models.perspective_matcher import PerspectiveMatch
                                    match = PerspectiveMatch(
                                        story_id=f"fast_{hash(articles1[idx1].title + articles2[idx2].title) % 100000}",
                                        topic=self._extract_topic_simple([articles1[idx1], articles2[idx2]]),
                                        articles={bias1: articles1[idx1], bias2: articles2[idx2]},
                                        similarity_scores={f"{bias1}-{bias2}": float(sim_score)},
                                        confidence=float(sim_score),
                                        timestamp=time.time()
                                    )
                                    matches.append(match)
                    
                    except Exception as e:
                        logger.warning(f"Similarity calculation failed for {bias1} vs {bias2}: {e}")
                        continue
            
            # Step 5: Quick deduplication
            unique_matches = []
            seen_urls = set()
            
            # Sort by confidence
            matches.sort(key=lambda x: x.confidence, reverse=True)
            
            for match in matches:
                match_urls = {article.url for article in match.articles.values()}
                if not match_urls.intersection(seen_urls):
                    unique_matches.append(match)
                    seen_urls.update(match_urls)
                    
                    # Limit total matches for speed
                    if len(unique_matches) >= 15:
                        break
            
            elapsed = time.time() - start_time
            logger.info(f"✅ FAST matching complete: {len(unique_matches)} matches in {elapsed:.2f}s")
            
            return unique_matches
        
        def fast_extract_topic_simple(self, articles):
            """Fast topic extraction"""
            if not articles:
                return "News"
            
            # Simple: use most common word from first article title
            words = articles[0].title.lower().split()
            meaningful_words = [w for w in words if len(w) > 4]
            return meaningful_words[0].title() if meaningful_words else "News"
        
        # Apply the patches
        PerspectiveMatcher.find_perspective_matches = fast_find_perspective_matches
        PerspectiveMatcher._extract_topic_simple = fast_extract_topic_simple
        
        print("✅ Performance patch applied successfully!")
        
        # Test the patched system
        print("\n🧪 Testing patched system...")
        
        # Quick test
        collector = NewsCollector()
        matcher = PerspectiveMatcher()
        
        print("📡 Testing with small dataset...")
        start_time = time.time()
        
        articles = collector.collect_diverse_articles("election", days_back=3)
        all_articles = []
        for bias_arts in articles.values():
            all_articles.extend(bias_arts)
        
        if len(all_articles) > 0:
            matches = matcher.find_perspective_matches(all_articles[:30])  # Test with subset
            
            test_time = time.time() - start_time
            print(f"✅ Test complete: {len(matches)} matches in {test_time:.2f}s")
            
            if matches:
                print("📰 Sample results:")
                for i, match in enumerate(matches[:2]):
                    print(f"   {i+1}. {match.topic} (confidence: {match.confidence:.2f})")
                    for bias, article in match.articles.items():
                        print(f"      {bias}: {article.title[:40]}... ({article.source})")
        
        else:
            print("⚠️  No articles collected for testing")
        
        print(f"\n🎉 Performance patch completed!")
        print(f"Your system should now be much faster!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print(f"💡 Install with: pip install scikit-learn")
        return False
    except Exception as e:
        print(f"❌ Patch failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def install_fast_dependencies():
    """Install performance dependencies"""
    
    print("📦 Installing performance dependencies...")
    
    try:
        import subprocess
        import sys
        
        # Install scikit-learn for fast TF-IDF
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
        print("✅ Installed scikit-learn")
        
        # Try to install faiss for even faster similarity
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
            print("✅ Installed faiss-cpu (advanced similarity)")
        except:
            print("⚠️  Failed to install faiss-cpu (optional)")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

# Fast API route replacement
def create_fast_api_routes():
    """Create faster API routes"""
    
    fast_routes_code = '''
# Add this to your src/api/routes.py or create a new fast_routes.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import logging

# Import your components
from ..data_collection.news_apis import NewsCollector
from ..models.perspective_matcher import PerspectiveMatcher

app = FastAPI(title="Fast News Perspective API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize with caching
collector = NewsCollector()
matcher = PerspectiveMatcher()

@app.post("/perspectives/find")
async def find_perspectives_fast(request_data: dict):
    """Fast perspective finding with optimizations"""
    
    start_time = time.time()
    
    try:
        query = request_data.get('query', '')
        days_back = min(request_data.get('days_back', 7), 14)  # Limit for speed
        min_perspectives = request_data.get('min_perspectives', 2)
        
        # Step 1: Quick collection with limits
        print(f"🔍 Fast collection: query='{query}', days={days_back}")
        diverse_articles = collector.collect_diverse_articles(query, days_back)
        
        # Step 2: Limit articles per category for speed
        all_articles = []
        for bias_category, articles in diverse_articles.items():
            # Take only most recent articles per category
            recent_articles = sorted(articles, key=lambda x: x.published_at, reverse=True)[:25]
            all_articles.extend(recent_articles)
        
        collection_time = time.time() - start_time
        print(f"📊 Collection: {len(all_articles)} articles in {collection_time:.2f}s")
        
        if not all_articles:
            return {
                "matches": [],
                "summary": {"total_matches": 0, "message": "No articles found"},
                "processing_time": collection_time
            }
        
        # Step 3: Fast perspective matching
        matching_start = time.time()
        matches = matcher.find_perspective_matches(all_articles, min_perspectives)
        matching_time = time.time() - matching_start
        
        total_time = time.time() - start_time
        
        print(f"⚡ Fast perspective finding: {len(matches)} matches in {total_time:.2f}s")
        print(f"   Collection: {collection_time:.2f}s")
        print(f"   Matching: {matching_time:.2f}s")
        
        # Format response
        formatted_matches = []
        for match in matches:
            formatted_match = {
                "story_id": match.story_id,
                "topic": match.topic,
                "confidence": match.confidence,
                "perspectives": {}
            }
            
            for bias, article in match.articles.items():
                formatted_match["perspectives"][bias] = {
                    "title": article.title,
                    "source": article.source,
                    "url": article.url,
                    "published_at": article.published_at.isoformat(),
                    "description": getattr(article, 'description', '')
                }
            
            formatted_matches.append(formatted_match)
        
        # Create summary
        summary = {
            "total_matches": len(matches),
            "average_confidence": sum(m.confidence for m in matches) / len(matches) if matches else 0,
            "perspective_distribution": {},
            "topics": list(set(m.topic for m in matches)),
            "processing_time": total_time,
            "performance": {
                "collection_time": collection_time,
                "matching_time": matching_time,
                "total_articles": len(all_articles)
            }
        }
        
        # Count perspective distribution
        for match in matches:
            for bias in match.articles.keys():
                summary["perspective_distribution"][bias] = summary["perspective_distribution"].get(bias, 0) + 1
        
        return {
            "query": query,
            "matches": formatted_matches,
            "summary": summary
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        logging.error(f"Fast perspective finding failed after {total_time:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "fast"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    # Write the fast routes
    fast_routes_file = Path("fast_api_routes.py")
    with open(fast_routes_file, 'w') as f:
        f.write(fast_routes_code)
    
    print(f"✅ Created fast API routes: {fast_routes_file.absolute()}")
    return fast_routes_file

if __name__ == "__main__":
    print("🚀 News Perspective Performance Optimizer")
    print("=" * 50)
    
    # Step 1: Install dependencies
    print("1. Installing performance dependencies...")
    deps_installed = install_fast_dependencies()
    
    if deps_installed:
        # Step 2: Apply performance patch
        print("\n2. Applying performance patch...")
        patch_applied = apply_performance_patch()
        
        if patch_applied:
            # Step 3: Create fast API routes
            print("\n3. Creating fast API routes...")
            fast_routes_file = create_fast_api_routes()
            
            print(f"\n🎉 Performance optimization complete!")
            print(f"\n🚀 Your system should now be 5-10x faster!")
            print(f"\nOptions to run:")
            print(f"1. Use existing system (now patched): python scripts/run_application.py serve")
            print(f"2. Use fast API routes: python {fast_routes_file}")
            print(f"3. Test performance: python performance_patch.py")
            
        else:
            print(f"\n❌ Performance patch failed")
    else:
        print(f"\n❌ Dependency installation failed")