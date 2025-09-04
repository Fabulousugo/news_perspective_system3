# tests/test_integration.py - Comprehensive Integration Testing
import pytest
import asyncio
import sys
from pathlib import Path
import tempfile
import time
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_collection.enhanced_news_collector import EnhancedNewsCollector
from models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
from models.bias_classifier import PoliticalBiasClassifier
from models.similarity_detector import SemanticSimilarityDetector

class TestSystemIntegration:
    """Comprehensive integration tests for the complete system"""
    
    @pytest.fixture
    async def news_collector(self):
        """Fixture providing configured news collector"""
        collector = EnhancedNewsCollector(
            config_path="config/news_sources.yaml",  # Use basic config for tests
            enable_caching=False  # Disable caching for consistent tests
        )
        async with collector:
            yield collector
    
    @pytest.fixture
    def perspective_matcher(self):
        """Fixture providing configured perspective matcher"""
        matcher = OptimizedPerspectiveMatcher(
            optimization_level="standard",  # Use standard for test consistency
            similarity_threshold=0.6  # Lower threshold for test data
        )
        return matcher
    
    @pytest.fixture
    def sample_articles(self):
        """Fixture providing sample articles for testing"""
        from data_collection.news_apis import Article
        from datetime import datetime
        
        return [
            Article(
                title="Economic Policy Changes Announced",
                content="The government announced new economic policies focusing on inflation control and job creation. The measures include tax reforms and infrastructure investment.",
                url="https://example.com/article1",
                source="Reuters",
                published_at=datetime.now(),
                description="Government announces new economic policies"
            ),
            Article(
                title="New Economic Measures Spark Debate",
                content="Critics argue the new economic policies favor wealthy individuals while supporters claim they will boost job growth nationwide.",
                url="https://example.com/article2",
                source="CNN",
                published_at=datetime.now(),
                description="Economic policy changes generate political debate"
            ),
            Article(
                title="Market Response to Policy Changes",
                content="Financial markets showed mixed reactions to the government's economic policy announcements, with some sectors gaining while others declined.",
                url="https://example.com/article3",
                source="Wall Street Journal",
                published_at=datetime.now(),
                description="Markets react to new economic policies"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_integration(self, perspective_matcher, sample_articles):
        """Test complete pipeline from articles to perspective matches"""
        
        # Test the complete pipeline
        matches = perspective_matcher.find_perspective_matches_fast(sample_articles)
        
        # Verify results
        assert isinstance(matches, list), "Should return list of matches"
        
        # Verify match structure if matches exist
        if matches:
            match = matches[0]
            assert hasattr(match, 'similarity_score'), "Match should have similarity score"
            assert hasattr(match, 'confidence'), "Match should have confidence"
            assert hasattr(match, 'articles'), "Match should have articles"
            
            assert 0 <= match.similarity_score <= 1, "Similarity should be in [0,1] range"
            assert 0 <= match.confidence <= 1, "Confidence should be in [0,1] range"
    
    def test_bias_classifier_integration(self, sample_articles):
        """Test bias classifier with sample articles"""
        
        classifier = PoliticalBiasClassifier(enable_ml_fallback=False)
        
        for article in sample_articles:
            result = classifier.classify_article_bias(article)
            
            assert hasattr(result, 'bias_label'), "Should have bias label"
            assert hasattr(result, 'confidence'), "Should have confidence"
            assert hasattr(result, 'method_used'), "Should have method used"
            assert hasattr(result, 'processing_time_ms'), "Should have processing time"
            
            assert result.bias_label in [0, 1, 2], "Bias label should be 0, 1, or 2"
            assert 0 <= result.confidence <= 1, "Confidence should be in [0,1] range"
            assert result.processing_time_ms >= 0, "Processing time should be non-negative"
    
    def test_similarity_detector_integration(self, sample_articles):
        """Test similarity detector with sample articles"""
        
        detector = SemanticSimilarityDetector(
            similarity_threshold=0.5,
            batch_size=16
        )
        
        # Test encoding
        embeddings = detector.encode_articles_batch(sample_articles)
        
        assert embeddings.shape[0] == len(sample_articles), "Should have one embedding per article"
        assert embeddings.shape[1] > 0, "Embeddings should have positive dimension"
        
        # Test similarity detection
        matches = detector.find_similar_articles(
            sample_articles[:2], 
            sample_articles[1:],
            cross_category_only=False
        )
        
        assert isinstance(matches, list), "Should return list of matches"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, perspective_matcher, sample_articles):
        """Test system performance meets minimum benchmarks"""
        
        # Duplicate sample articles to create larger test set
        test_articles = sample_articles * 10  # 30 articles total
        
        start_time = time.perf_counter()
        matches = perspective_matcher.find_perspective_matches_fast(test_articles)
        processing_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Performance benchmarks
        articles_per_second = len(test_articles) / (processing_time / 1000)
        
        # Minimum performance requirements
        assert articles_per_second > 10, f"Should process >10 articles/second, got {articles_per_second:.1f}"
        assert processing_time < 5000, f"Should complete in <5s, took {processing_time:.1f}ms"
        
        # Get performance statistics
        stats = perspective_matcher.get_performance_statistics()
        assert 'articles_processed' in stats, "Should track articles processed"
        assert stats['articles_processed'] >= len(test_articles), "Should count processed articles"
    
    @pytest.mark.asyncio
    async def test_error_handling_resilience(self, perspective_matcher):
        """Test system resilience to various error conditions"""
        
        # Test empty input
        matches = perspective_matcher.find_perspective_matches_fast([])
        assert matches == [], "Should handle empty input gracefully"
        
        # Test articles with missing fields
        from data_collection.news_apis import Article
        from datetime import datetime
        
        incomplete_articles = [
            Article(
                title="",  # Empty title
                content="Some content here",
                url="https://example.com/test",
                source="Test Source",
                published_at=datetime.now()
            )
        ]
        
        # Should not raise exception
        try:
            matches = perspective_matcher.find_perspective_matches_fast(incomplete_articles)
            assert isinstance(matches, list), "Should return list even with problematic input"
        except ValueError:
            # Article validation might reject incomplete articles, which is acceptable
            pass
    
    def test_configuration_validation(self):
        """Test system handles configuration errors gracefully"""
        
        # Test with non-existent config file
        try:
            collector = EnhancedNewsCollector(
                config_path="non_existent_config.yaml"
            )
        except Exception as e:
            assert "Failed to load source config" in str(e) or "No such file" in str(e)
    
    def test_memory_usage_constraints(self, perspective_matcher, sample_articles):
        """Test system operates within memory constraints"""
        
        import psutil
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process articles multiple times to test memory leaks
        for _ in range(10):
            matches = perspective_matcher.find_perspective_matches_fast(sample_articles)
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for small test)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB, possible memory leak"
    
    def test_concurrent_processing_safety(self, perspective_matcher, sample_articles):
        """Test system handles concurrent processing safely"""
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def process_articles():
            try:
                matches = perspective_matcher.find_perspective_matches_fast(sample_articles)
                results_queue.put(matches)
            except Exception as e:
                errors_queue.put(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=process_articles)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Check for errors
        assert errors_queue.empty(), f"Concurrent processing errors: {list(errors_queue.queue)}"
        
        # Verify all threads completed successfully
        assert results_queue.qsize() == 5, "All threads should complete successfully"

class TestPerformanceRegression:
    """Performance regression tests to ensure optimizations don't degrade"""
    
    PERFORMANCE_BASELINES = {
        'articles_per_second_min': 20.0,
        'max_processing_time_ms': 10000,
        'max_memory_mb': 300,
        'min_similarity_threshold': 0.5
    }
    
    @pytest.fixture
    def performance_test_articles(self):
        """Generate larger set of articles for performance testing"""
        from data_collection.news_apis import Article
        from datetime import datetime
        
        articles = []
        for i in range(50):  # 50 test articles
            articles.append(Article(
                title=f"Test Article {i} - Economic Policy Discussion",
                content=f"This is test article number {i} discussing various economic policies and their implications for the economy. " * 5,
                url=f"https://example.com/article{i}",
                source=["Reuters", "CNN", "Fox News", "BBC News"][i % 4],
                published_at=datetime.now(),
                description=f"Test article {i} for performance testing"
            ))
        
        return articles
    
    def test_processing_speed_regression(self, performance_test_articles):
        """Test processing speed meets baseline requirements"""
        
        matcher = OptimizedPerspectiveMatcher(
            optimization_level="quantized",
            similarity_threshold=0.65
        )
        
        start_time = time.perf_counter()
        matches = matcher.find_perspective_matches_fast(performance_test_articles)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        articles_per_second = len(performance_test_articles) / (processing_time_ms / 1000)
        
        assert articles_per_second >= self.PERFORMANCE_BASELINES['articles_per_second_min'], \
            f"Processing speed regression: {articles_per_second:.1f} < {self.PERFORMANCE_BASELINES['articles_per_second_min']}"
        
        assert processing_time_ms <= self.PERFORMANCE_BASELINES['max_processing_time_ms'], \
            f"Processing time regression: {processing_time_ms:.1f}ms > {self.PERFORMANCE_BASELINES['max_processing_time_ms']}ms"
    
    def test_memory_usage_regression(self, performance_test_articles):
        """Test memory usage stays within acceptable bounds"""
        
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        matcher = OptimizedPerspectiveMatcher(
            optimization_level="quantized",
            similarity_threshold=0.65
        )
        
        # Process articles multiple times
        for _ in range(5):
            matches = matcher.find_perspective_matches_fast(performance_test_articles)
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_used = peak_memory - initial_memory
        
        assert memory_used <= self.PERFORMANCE_BASELINES['max_memory_mb'], \
            f"Memory usage regression: {memory_used:.1f}MB > {self.PERFORMANCE_BASELINES['max_memory_mb']}MB"

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
