# scripts/setup_speed_optimizations.py - Setup script for model optimizations

import subprocess
import sys,os
import time
from pathlib import Path
import logging

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def install_optimization_dependencies():
    """Install required packages for speed optimizations"""
    print("📦 Installing Speed Optimization Dependencies")
    print("=" * 60)
    
    # Required packages for optimizations
    optimization_packages = [
        "torch",  # Ensure latest torch for quantization
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0", 
        "onnxruntime-tools",  # For optimization tools
        "numpy>=1.21.0",
    ]
    
    print("Installing optimization packages...")
    
    for package in optimization_packages:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("\n✅ All optimization dependencies installed!")
    return True

def create_optimized_models():
    """Create all optimized model versions"""
    print("\n🚀 Creating Optimized Models")
    print("=" * 40)
    
    try:
        from src.models.optimized_models import ModelOptimizer
        
        print("Creating quantized and ONNX versions of models...")
        ModelOptimizer.optimize_all_models()
        
        print("✅ All optimized models created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create optimized models: {e}")
        print("This is normal if you don't have pre-trained models yet.")
        return False

def benchmark_performance():
    """Benchmark different optimization levels"""
    print("\n🧪 Benchmarking Performance")
    print("=" * 30)
    
    try:
        from src.models.optimized_models import ModelOptimizer
        
        # Create test articles
        test_texts = [
            "This progressive policy will help working families across America.",
            "Conservative fiscal responsibility is essential for economic growth.",
            "Bipartisan cooperation is needed to address climate change challenges.",
            "The liberal agenda threatens traditional American values.",
            "Centrist approaches often find the best solutions to complex problems."
        ] * 6  # 30 test articles
        
        print(f"Testing with {len(test_texts)} articles...")
        
        results = ModelOptimizer.benchmark_models(test_texts, iterations=2)
        
        print("\n📊 Performance Results:")
        print("-" * 60)
        for opt_level, metrics in results.items():
            if 'error' in metrics:
                print(f"❌ {opt_level}: {metrics['error']}")
            else:
                print(f"✅ {opt_level.upper()}")
                print(f"   Throughput: {metrics['throughput_articles_per_second']:.1f} articles/sec")
                print(f"   Time: {metrics['avg_time_seconds']:.3f}s")
                print(f"   Speedup: {metrics['speedup_vs_standard']:.1f}x")
        
        # Show model sizes
        sizes = ModelOptimizer.get_model_sizes()
        if sizes:
            print(f"\n💾 Model Sizes:")
            for model_type, size in sizes.items():
                print(f"   {model_type}: {size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_optimized_system():
    """Test the optimized perspective matching system"""
    print("\n🎯 Testing Optimized Perspective System")
    print("=" * 45)
    
    try:
        from src.models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
        from src.data_collection.news_apis import Article
        from datetime import datetime
        
        # Create test articles
        test_articles = [
            Article("CNN reports on climate policy", "Progressive climate action needed", "url1", "CNN", datetime.now(), bias_label=0),
            Article("Fox News critiques environmental rules", "New regulations hurt business", "url2", "Fox News", datetime.now(), bias_label=2),
            Article("Reuters covers climate debate", "Mixed reactions to climate policy", "url3", "Reuters", datetime.now(), bias_label=1),
            Article("Breitbart opposes green policies", "Climate agenda threatens economy", "url4", "Breitbart", datetime.now(), bias_label=2),
            Article("Guardian supports climate action", "Urgent action needed on climate", "url5", "The Guardian", datetime.now(), bias_label=0),
        ]
        
        print(f"Testing with {len(test_articles)} sample articles...")
        
        # Test different optimization levels
        optimization_levels = ["quantized", "onnx"]
        
        for opt_level in optimization_levels:
            print(f"\n🧪 Testing {opt_level} optimization...")
            
            matcher = OptimizedPerspectiveMatcher(optimization_level=opt_level)
            
            start_time = time.time()
            matches = matcher.find_perspective_matches_fast(test_articles)
            end_time = time.time()
            
            stats = matcher.get_performance_stats()
            
            print(f"✅ {opt_level.upper()} Results:")
            print(f"   Processing time: {end_time - start_time:.3f}s")
            print(f"   Matches found: {len(matches)}")
            print(f"   Articles processed: {stats['total_articles_processed']}")
            
            if matches:
                print(f"   Sample matches:")
                for i, match in enumerate(matches[:2]):
                    print(f"      {i+1}. {match.topic} (confidence: {match.confidence:.3f})")
                    for bias, article in match.articles.items():
                        print(f"         {bias}: {article.source}")
        
        print("\n✅ Optimized system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main setup function"""
    print("🚀 Speed Optimization Setup for News Perspective System")
    print("=" * 70)
    print("This will set up quantization and ONNX optimizations for faster inference")
    print("")
    
    # Step 1: Install dependencies
    if not install_optimization_dependencies():
        print("❌ Failed to install dependencies. Exiting.")
        return False
    
    # Step 2: Create optimized models (optional - may fail without pre-trained models)
    create_optimized_models()
    
    # Step 3: Benchmark performance
    benchmark_performance()
    
    # Step 4: Test optimized system
    test_optimized_system()
    
    print("\n🎉 Speed Optimization Setup Complete!")
    print("\n📊 Expected Performance Improvements:")
    print("   🔹 Quantized models: 2-4x faster on CPU")
    print("   🔹 ONNX models: 3-6x faster on CPU")
    print("   🔹 Reduced memory usage: 50-75% less RAM")
    print("   🔹 Smaller model files: 25-50% size reduction")
    
    print("\n🚀 Next Steps:")
    print("   1. Use optimized browser:")
    print("      python scripts/speed_optimized_browser.py browse --optimization onnx")
    print("   2. Compare performance:")
    print("      python scripts/speed_optimized_browser.py benchmark")
    print("   3. Configure optimization level:")
    print("      python scripts/speed_optimized_browser.py configure --optimization quantized")
    
    return True

if __name__ == "__main__":
    main()

