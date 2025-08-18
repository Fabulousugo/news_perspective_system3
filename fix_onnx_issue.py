# fix_onnx_issue.py - Fix ONNX export issues and test optimizations

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_onnx_compatibility():
    """Check ONNX compatibility and versions"""
    print("🔍 Checking ONNX Compatibility")
    print("=" * 35)
    
    try:
        import torch
        import onnx
        import onnxruntime as ort
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ ONNX version: {onnx.__version__}")
        print(f"✅ ONNX Runtime version: {ort.__version__}")
        
        # Check ONNX Runtime providers
        providers = ort.get_available_providers()
        print(f"✅ Available ONNX providers: {providers}")
        
        # Check opset versions
        print(f"✅ ONNX supported opsets: {onnx.defs.onnx_opset_version()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_simple_onnx_export():
    """Test ONNX export with a simple model"""
    print("\n🧪 Testing Simple ONNX Export")
    print("=" * 32)
    
    try:
        import torch
        import torch.nn as nn
        from pathlib import Path
        
        # Create a simple test model
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 3)
                
            def forward(self, x):
                return self.linear(x)
        
        # Test export
        model = SimpleTestModel()
        model.eval()
        
        dummy_input = torch.randn(1, 10)
        test_path = Path("test_simple_model.onnx")
        
        # Try different opset versions
        for opset in [17, 16, 15, 14]:
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(test_path),
                    opset_version=opset,
                    input_names=['input'],
                    output_names=['output']
                )
                
                print(f"✅ Simple ONNX export successful with opset {opset}")
                
                # Clean up
                if test_path.exists():
                    test_path.unlink()
                
                return opset
                
            except Exception as e:
                print(f"❌ Opset {opset} failed: {e}")
                continue
        
        print("❌ All opset versions failed for simple model")
        return None
        
    except Exception as e:
        print(f"❌ Simple ONNX export test failed: {e}")
        return None

def test_quantized_models():
    """Test that quantized models work properly"""
    print("\n⚡ Testing Quantized Models")
    print("=" * 28)
    
    try:
        from models.optimized_models import OptimizedBiasClassifier
        
        print("🔧 Initializing quantized model...")
        classifier = OptimizedBiasClassifier(optimization_level="quantized")
        
        print("🧪 Testing prediction...")
        test_text = "This is a test sentence for political bias classification."
        prediction = classifier.predict_single(test_text)
        
        print(f"✅ Quantized model working!")
        print(f"   Prediction: {prediction['predicted_class']}")
        print(f"   Confidence: {prediction['confidence']:.3f}")
        
        # Test batch prediction
        print("🧪 Testing batch prediction...")
        test_texts = [
            "Progressive climate policies are essential.",
            "Conservative fiscal responsibility matters.",
            "Bipartisan cooperation is needed."
        ]
        
        import time
        start_time = time.time()
        predictions = classifier.predict(test_texts)
        end_time = time.time()
        
        print(f"✅ Batch prediction successful!")
        print(f"   Processed {len(test_texts)} texts in {end_time - start_time:.3f}s")
        print(f"   Throughput: {len(test_texts)/(end_time - start_time):.1f} texts/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantized model test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def create_robust_optimizer():
    """Create a version that gracefully handles ONNX failures"""
    print("\n🛠️  Creating Robust Optimization System")
    print("=" * 42)
    
    robust_code = '''# src/models/robust_optimizer.py - ONNX-safe optimization

import logging
from typing import List, Dict, Optional
from .optimized_models import OptimizedBiasClassifier, OptimizedSimilarityDetector

logger = logging.getLogger(__name__)

class RobustOptimizer:
    """
    Robust optimizer that gracefully handles ONNX failures
    """
    
    def __init__(self, preferred_optimization: str = "quantized"):
        """
        Initialize with fallback strategy
        
        Args:
            preferred_optimization: Preferred level, falls back if needed
        """
        self.preferred_optimization = preferred_optimization
        self.actual_optimization = None
        self.bias_classifier = None
        self.similarity_detector = None
        
        self._initialize_with_fallback()
    
    def _initialize_with_fallback(self):
        """Initialize with automatic fallback"""
        # Try optimization levels in order of preference
        optimization_order = []
        
        if self.preferred_optimization == "onnx":
            optimization_order = ["onnx", "quantized", "standard"]
        elif self.preferred_optimization == "quantized":
            optimization_order = ["quantized", "standard"]
        else:
            optimization_order = ["standard"]
        
        for opt_level in optimization_order:
            try:
                logger.info(f"🔧 Trying {opt_level} optimization...")
                
                # Test bias classifier
                test_classifier = OptimizedBiasClassifier(optimization_level=opt_level)
                test_classifier.predict_single("Test sentence")
                
                # Test similarity detector  
                similarity_opt = "onnx" if opt_level == "onnx" else "standard"
                test_similarity = OptimizedSimilarityDetector(optimization_level=similarity_opt)
                test_similarity.encode_articles(["Test article"])
                
                # If we get here, it worked
                self.actual_optimization = opt_level
                self.bias_classifier = test_classifier
                self.similarity_detector = test_similarity
                
                logger.info(f"✅ Successfully initialized with {opt_level} optimization")
                break
                
            except Exception as e:
                logger.warning(f"❌ {opt_level} optimization failed: {e}")
                continue
        
        if self.actual_optimization is None:
            raise RuntimeError("All optimization levels failed!")
    
    def get_optimization_info(self) -> Dict:
        """Get information about the current optimization"""
        return {
            'preferred': self.preferred_optimization,
            'actual': self.actual_optimization,
            'fallback_occurred': self.preferred_optimization != self.actual_optimization,
            'available_optimizations': self._get_available_optimizations()
        }
    
    def _get_available_optimizations(self) -> List[str]:
        """Test which optimizations are available"""
        available = []
        
        for opt_level in ["standard", "quantized", "onnx"]:
            try:
                test_classifier = OptimizedBiasClassifier(optimization_level=opt_level)
                test_classifier.predict_single("Test")
                available.append(opt_level)
            except:
                pass
        
        return available

# Example usage
if __name__ == "__main__":
    optimizer = RobustOptimizer(preferred_optimization="onnx")
    info = optimizer.get_optimization_info()
    print(f"Optimization info: {info}")
'''
    
    # Write the robust optimizer
    robust_path = Path("src/models/robust_optimizer.py")
    robust_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(robust_path, 'w') as f:
        f.write(robust_code)
    
    print(f"✅ Created robust optimizer: {robust_path}")
    return True

def fix_onnx_issues():
    """Main fix function"""
    print("🔧 ONNX Issue Fix and Optimization Test")
    print("=" * 45)
    print("This will diagnose ONNX issues and set up reliable fallbacks")
    print("")
    
    # Step 1: Check compatibility
    compat_ok = check_onnx_compatibility()
    
    # Step 2: Test simple ONNX export
    working_opset = test_simple_onnx_export()
    
    # Step 3: Test quantized models (should always work)
    quantized_ok = test_quantized_models()
    
    # Step 4: Create robust optimizer
    robust_ok = create_robust_optimizer()
    
    # Summary and recommendations
    print("\n📊 Diagnosis Summary")
    print("=" * 25)
    
    print(f"🔍 ONNX compatibility: {'✅ OK' if compat_ok else '❌ Issues'}")
    print(f"🧪 Simple ONNX export: {'✅ Works' if working_opset else '❌ Failed'}")
    print(f"⚡ Quantized models: {'✅ Working' if quantized_ok else '❌ Failed'}")
    print(f"🛠️  Robust fallback: {'✅ Created' if robust_ok else '❌ Failed'}")
    
    if working_opset:
        print(f"💡 Working ONNX opset: {working_opset}")
    
    print("\n🚀 Recommendations:")
    
    if quantized_ok:
        print("✅ Use quantized optimization (reliable, 2-4x speedup):")
        print("   python scripts/speed_optimized_browser.py browse --optimization quantized")
        
        if working_opset:
            print("✅ ONNX may work with manual setup:")
            print(f"   Try updating the opset version to {working_opset} in the code")
        else:
            print("⚠️  ONNX optimization has compatibility issues")
            print("   Stick with quantized optimization for reliability")
        
        print("\n🎯 Test the working system:")
        print("   python test_speed_optimizations.py")
        
        return True
    else:
        print("❌ Quantized models failed - check your PyTorch installation")
        print("💡 Fallback to standard models:")
        print("   python scripts/simple_enhanced_browser.py browse")
        return False

if __name__ == "__main__":
    success = fix_onnx_issues()
    print(f"\n{'🎉 Fix completed!' if success else '❌ Issues remain - check PyTorch installation'}")