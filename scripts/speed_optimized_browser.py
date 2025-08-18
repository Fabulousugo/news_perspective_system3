# scripts/speed_optimized_browser.py - CLI for optimized news browsing

import click
import logging
import time
from pathlib import Path
import sys,os
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.simple_extended_collector import SimpleExtendedCollector
from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher, SpeedBenchmark
from src.models.optimized_models import ModelOptimizer
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """🚀 Speed-Optimized News Perspective Browser"""
    pass

@cli.command()
@click.option('--query', '-q', default='', help='Search query')
@click.option('--days', '-d', default=7, help='Days to look back')
@click.option('--optimization', '-o', type=click.Choice(['standard', 'quantized', 'onnx', 'auto']), 
              default='auto', help='Optimization level (auto = best available)')
@click.option('--limit', '-l', default=20, help='Number of articles to show')
def browse(query: str, days: int, optimization: str, limit: int):
    """Browse news with speed-optimized models"""
    
    print(f"🚀 Speed-Optimized News Perspective Browser")
    print("=" * 60)
    print(f"🔍 Query: {query or 'General news'}")
    print(f"📅 Days back: {days}")
    print(f"⚡ Optimization: {optimization}")
    print("")
    
    try:
        # Initialize components with optimization
        print(f"🔧 Initializing with {optimization} optimization...")
        collector = SimpleExtendedCollector()
        
        # Handle auto optimization selection
        if optimization == 'auto':
            # Try optimizations in order of preference
            for opt_level in ['onnx', 'quantized', 'standard']:
                try:
                    print(f"   Trying {opt_level}...")
                    test_matcher = OptimizedPerspectiveMatcher(optimization_level=opt_level)
                    # Test that it works
                    test_matcher.get_performance_stats()
                    optimization = opt_level
                    matcher = test_matcher
                    print(f"   ✅ Using {opt_level} optimization")
                    break
                except Exception as e:
                    print(f"   ❌ {opt_level} failed: {str(e)[:60]}...")
                    continue
            else:
                raise RuntimeError("All optimization levels failed")
        else:
            # Use specified optimization with fallback handling
            try:
                matcher = OptimizedPerspectiveMatcher(optimization_level=optimization)
                actual_opt = matcher.optimization_level
                if actual_opt != optimization:
                    print(f"   ⚠️  Fallback: Using {actual_opt} instead of {optimization}")
                    optimization = actual_opt
            except Exception as e:
                print(f"   ❌ {optimization} failed: {e}")
                print(f"   🔄 Falling back to quantized...")
                matcher = OptimizedPerspectiveMatcher(optimization_level='quantized')
                optimization = 'quantized'
        
        # Collect articles
        print("📡 Collecting articles...")
        start_collection = time.time()
        
        diverse_articles = collector.collect_diverse_articles(
            query=query,
            days_back=days
        )
        
        collection_time = time.time() - start_collection
        
        # Flatten articles
        all_articles = []
        for bias_articles in diverse_articles.values():
            all_articles.extend(bias_articles)
        
        print(f"✅ Collected {len(all_articles)} articles in {collection_time:.2f}s")
        
        if len(all_articles) == 0:
            print("❌ No articles found. Check your API keys and query.")
            return
        
        # Fast perspective matching
        print(f"\n⚡ Finding perspectives with {optimization} optimization...")
        start_matching = time.time()
        
        matches = matcher.find_perspective_matches_fast(all_articles)
        
        matching_time = time.time() - start_matching
        total_time = collection_time + matching_time
        
        # Get performance stats
        perf_stats = matcher.get_performance_stats()
        
        print(f"\n📊 Performance Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Collection time: {collection_time:.2f}s")
        print(f"   Matching time: {matching_time:.2f}s")
        print(f"   Articles processed: {perf_stats['total_articles_processed']}")
        print(f"   Processing speed: {perf_stats['total_articles_processed']/total_time:.1f} articles/sec")
        print(f"   Perspective matches: {len(matches)}")
        print(f"   Optimization used: {optimization}")
        
        # Show expected speedup
        speedup_estimates = {'standard': '1x', 'quantized': '2-4x', 'onnx': '3-6x'}
        expected_speedup = speedup_estimates.get(optimization, '1x')
        print(f"   Expected speedup: {expected_speedup} vs standard")
        
        # Show matches
        if matches:
            print(f"\n🎯 Found {len(matches)} perspective matches:")
            print("=" * 80)
            
            bias_names = {0: "🔵 LEFT", 1: "⚪ CENTER", 2: "🔴 RIGHT"}
            
            for i, match in enumerate(matches[:limit]):
                print(f"\n[{i+1:2d}] Topic: {match.topic} | Confidence: {match.confidence:.3f}")
                
                for bias_category, article in match.articles.items():
                    bias_icon = bias_names.get(article.bias_label, "❓")
                    time_str = article.published_at.strftime("%m/%d %H:%M")
                    
                    print(f"     {bias_icon} {bias_category.upper()} | {time_str}")
                    print(f"       📰 {article.title}")
                    print(f"       🏢 {article.source}")
                    print(f"       🔗 {article.url}")
                
                print("     " + "─" * 70)
            
            # Performance tip
            if optimization != 'onnx':
                print(f"\n💡 Tip: Try --optimization auto for best available speed!")
            
        else:
            print(f"\n❌ No perspective matches found")
            print(f"💡 Try:")
            print(f"   • Broader search query")
            print(f"   • More days (--days 14)")
            print(f"   • Lower similarity threshold")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Optimized browse failed: {e}")
        import traceback
        print(f"Debug: {traceback.format_exc()}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Try: python fix_onnx_issue.py")
        print(f"   2. Use quantized: --optimization quantized")
        print(f"   3. Fallback to: python scripts/simple_enhanced_browser.py browse")

@cli.command()
@click.option('--articles', '-a', default=50, help='Number of test articles')
def benchmark(articles: int):
    """Benchmark different optimization levels"""
    
    print("🧪 Performance Benchmark")
    print("=" * 30)
    print(f"Testing with {articles} articles")
    print("")
    
    try:
        # Collect test articles
        print("📡 Collecting test articles...")
        collector = SimpleExtendedCollector()
        diverse_articles = collector.collect_diverse_articles("", days_back=7)
        
        all_articles = []
        for bias_articles in diverse_articles.values():
            all_articles.extend(bias_articles)
        
        test_articles = all_articles[:articles]
        print(f"✅ Using {len(test_articles)} articles for benchmark")
        
        if len(test_articles) < 10:
            print("❌ Not enough articles for meaningful benchmark")
            return
        
        # Benchmark perspective matching
        print(f"\n🏃 Benchmarking perspective matching...")
        results = SpeedBenchmark.benchmark_perspective_matching(test_articles, iterations=2)
        
        print(f"\n📊 Benchmark Results:")
        print("=" * 60)
        
        for opt_level, metrics in results.items():
            if 'error' in metrics:
                print(f"❌ {opt_level.upper()}: {metrics['error']}")
            else:
                print(f"✅ {opt_level.upper()}")
                print(f"   Time: {metrics['avg_time_seconds']:.3f}s")
                print(f"   Throughput: {metrics['throughput_articles_per_second']:.1f} articles/sec")
                print(f"   Speedup: {metrics.get('speedup_vs_standard', 1.0):.1f}x")
                if 'matches_found' in metrics:
                    print(f"   Matches found: {metrics['matches_found']}")
                print("")
        
        # Model benchmark
        print(f"🧪 Benchmarking bias classification models...")
        test_texts = [a.title + ". " + (a.description or "") for a in test_articles[:30]]
        
        model_results = ModelOptimizer.benchmark_models(test_texts, iterations=2)
        
        print(f"\n📊 Model Performance:")
        print("=" * 40)
        
        for opt_level, metrics in model_results.items():
            if 'error' in metrics:
                print(f"❌ {opt_level.upper()}: {metrics['error']}")
            else:
                print(f"✅ {opt_level.upper()}")
                print(f"   Throughput: {metrics['throughput_articles_per_second']:.1f} articles/sec")
                print(f"   Speedup: {metrics.get('speedup_vs_standard', 1.0):.1f}x")
        
        # Show model sizes
        sizes = ModelOptimizer.get_model_sizes()
        if sizes:
            print(f"\n💾 Model Sizes:")
            for model_type, size in sizes.items():
                print(f"   {model_type}: {size}")
        
        print(f"\n💡 Recommendations:")
        print(f"   🔹 For fastest processing: use --optimization onnx")
        print(f"   🔹 For balanced speed/setup: use --optimization quantized")
        print(f"   🔹 For maximum compatibility: use --optimization standard")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        print(traceback.format_exc())

@cli.command()
@click.option('--optimization', '-o', type=click.Choice(['standard', 'quantized', 'onnx']), 
              default='quantized', help='Set default optimization level')
@click.option('--threshold', '-t', default=0.65, help='Similarity threshold')
def configure(optimization: str, threshold: float):
    """Configure speed optimization settings"""
    
    print("⚙️ Speed Optimization Configuration")
    print("=" * 40)
    
    print(f"📊 Settings:")
    print(f"   Optimization level: {optimization}")
    print(f"   Similarity threshold: {threshold}")
    print("")
    
    # Update settings
    settings.SIMILARITY_THRESHOLD = threshold
    
    print(f"✅ Configuration updated!")
    print("")
    
    print(f"📊 Optimization Levels:")
    print(f"   🔹 standard: Original models (slower, highest compatibility)")
    print(f"   🔹 quantized: 8-bit quantized models (2-4x faster)")  
    print(f"   🔹 onnx: ONNX Runtime optimized (3-6x faster)")
    print("")
    
    print(f"💡 Performance Tips:")
    print(f"   • Lower similarity threshold = more matches")
    print(f"   • Higher threshold = fewer but more precise matches") 
    print(f"   • ONNX optimization works best on CPU")
    print(f"   • Quantized models use less memory")

@cli.command()
def setup():
    """Set up speed optimizations"""
    
    print("🚀 Setting up Speed Optimizations")
    print("=" * 40)
    
    try:
        # Import and run setup
        from setup_speed_optimizations import main as setup_main
        setup_main()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\nManual setup:")
        print("1. pip install onnx onnxruntime")
        print("2. python scripts/setup_speed_optimizations.py")

if __name__ == "__main__":
    cli()