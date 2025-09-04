# scripts/setup_production.py - Production Deployment Setup
import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
import logging

class ProductionSetup:
    """Production deployment setup and configuration"""
    
    def __init__(self, deployment_type: str = "standard"):
        self.deployment_type = deployment_type
        self.project_root = Path(__file__).parent.parent
        self.logger = self._setup_logging()
        
        # Deployment configurations
        self.deployments = {
            "standard": {
                "optimization_level": "quantized",
                "memory_limit_mb": 512,
                "workers": 1,
                "port": 8000
            },
            "high_performance": {
                "optimization_level": "onnx",
                "memory_limit_mb": 1024,
                "workers": 2,
                "port": 8001
            },
            "development": {
                "optimization_level": "standard", 
                "memory_limit_mb": 256,
                "workers": 1,
                "port": 8000
            }
        }
    
    def _setup_logging(self):
        """Setup logging for deployment process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def setup_environment(self):
        """Setup complete production environment"""
        
        self.logger.info(f"Setting up {self.deployment_type} deployment environment...")
        
        try:
            # Create necessary directories
            self._create_directories()
            
            # Install dependencies
            self._install_dependencies()
            
            # Setup configuration files
            self._setup_configuration()
            
            # Initialize data directories
            self._initialize_data_directories()
            
            # Download required models
            self._download_models()
            
            # Setup systemd service (Linux only)
            if self._is_linux():
                self._setup_systemd_service()
            
            # Create startup scripts
            self._create_startup_scripts()
            
            # Validate setup
            self._validate_setup()
            
            self.logger.info("Production setup completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            raise
    
    def _create_directories(self):
        """Create necessary directories for production deployment"""
        
        directories = [
            "data/raw",
            "data/processed", 
            "data/cache",
            "data/models",
            "logs",
            "config",
            "scripts"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
    
    def _install_dependencies(self):
        """Install production dependencies"""
        
        self.logger.info("Installing production dependencies...")
        
        # Install core dependencies
        requirements = [
            "torch>=1.9.0",
            "transformers>=4.20.0", 
            "sentence-transformers>=2.2.0",
            "scikit-learn>=1.1.0",
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "aiohttp>=3.8.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "pyyaml>=6.0",
            "feedparser>=6.0.8",
            "beautifulsoup4>=4.9.0",
            "nltk>=3.6.0",
            "psutil>=5.8.0"
        ]
        
        # Add ONNX dependencies for high performance deployment
        if self.deployment_type == "high_performance":
            requirements.extend([
                "onnx>=1.12.0",
                "onnxruntime>=1.12.0"
            ])
        
        for requirement in requirements:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", requirement
                ], check=True, capture_output=True)
                self.logger.info(f"Installed: {requirement}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to install {requirement}: {e}")
                raise
    
    def _setup_configuration(self):
        """Setup production configuration files"""
        
        config = self.deployments[self.deployment_type]
        
        # Production settings
        settings_config = {
            "BIAS_CLASSIFIER_MODEL": "distilbert-base-uncased",
            "SIMILARITY_MODEL": "all-MiniLM-L6-v2", 
            "SIMILARITY_THRESHOLD": 0.65,
            "BATCH_SIZE": 16,
            "MAX_SEQUENCE_LENGTH": 512,
            "ENABLE_MODEL_QUANTIZATION": config["optimization_level"] in ["quantized", "onnx"],
            "ENABLE_ONNX_OPTIMIZATION": config["optimization_level"] == "onnx",
            "MEMORY_LIMIT_MB": config["memory_limit_mb"],
            "CPU_THREADS": 4,
            "CACHE_TTL_SECONDS": 3600
        }
        
        # Write production settings
        settings_file = self.project_root / "config" / "production_settings.py"
        with open(settings_file, 'w') as f:
            f.write("# Production Settings - Auto-generated\n")
            f.write("from pydantic import BaseSettings, Field\n\n")
            f.write("class ProductionSettings(BaseSettings):\n")
            for key, value in settings_config.items():
                if isinstance(value, str):
                    f.write(f'    {key}: str = "{value}"\n')
                else:
                    f.write(f'    {key}: {type(value).__name__} = {value}\n')
        
        self.logger.info(f"Created production settings: {settings_file}")
        
        # Environment file template
        env_file = self.project_root / ".env.production"
        with open(env_file, 'w') as f:
            f.write("# Production Environment Variables\n")
            f.write("# Copy to .env and fill in your API keys\n\n")
            f.write("NEWS_API_KEY=your_news_api_key_here\n")
            f.write("GUARDIAN_API_KEY=your_guardian_api_key_here\n")
            f.write(f"OPTIMIZATION_LEVEL={config['optimization_level']}\n")
            f.write(f"SERVER_PORT={config['port']}\n")
            f.write(f"WORKERS={config['workers']}\n")
        
        self.logger.info(f"Created environment template: {env_file}")
    
    def _initialize_data_directories(self):
        """Initialize data directories with proper permissions"""
        
        # Set up cache directories
        cache_dirs = ["data/cache/embeddings", "data/cache/articles", "data/cache/models"]
        for cache_dir in cache_dirs:
            dir_path = self.project_root / cache_dir
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Set permissions (Unix-like systems)
            if hasattr(os, 'chmod'):
                os.chmod(dir_path, 0o755)
        
        # Create log files
        log_files = ["logs/application.log", "logs/performance.log", "logs/errors.log"]
        for log_file in log_files:
            log_path = self.project_root / log_file
            log_path.touch()
            
            if hasattr(os, 'chmod'):
                os.chmod(log_path, 0o644)
        
        self.logger.info("Initialized data directories")
    
    def _download_models(self):
        """Download and cache required models"""
        
        self.logger.info("Downloading required models...")
        
        try:
            # Download sentence transformer model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            model_path = self.project_root / "data" / "models" / "sentence_transformer"
            model.save(str(model_path))
            self.logger.info("Downloaded sentence transformer model")
            
            # Download DistilBERT model
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=3
            )
            
            bert_path = self.project_root / "data" / "models" / "distilbert"
            bert_path.mkdir(exist_ok=True)
            tokenizer.save_pretrained(str(bert_path))
            model.save_pretrained(str(bert_path))
            self.logger.info("Downloaded DistilBERT model")
            
        except Exception as e:
            self.logger.error(f"Model download failed: {e}")
            raise
    
    def _is_linux(self):
        """Check if running on Linux (for systemd setup)"""
        return sys.platform.startswith('linux')
    
    def _setup_systemd_service(self):
        """Setup systemd service for automatic startup (Linux)"""
        
        if not self._is_linux():
            return
        
        config = self.deployments[self.deployment_type]
        
        service_content = f"""[Unit]
Description=News Perspective Diversification Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={self.project_root}
Environment=PATH={os.environ.get('PATH', '')}
ExecStart={sys.executable} -m uvicorn src.api.routes:app --host 0.0.0.0 --port {config['port']} --workers {config['workers']}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.project_root / "scripts" / "news-perspective.service"
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        self.logger.info(f"Created systemd service file: {service_file}")
        self.logger.info("To install: sudo cp scripts/news-perspective.service /etc/systemd/system/")
        self.logger.info("To enable: sudo systemctl enable news-perspective")
        self.logger.info("To start: sudo systemctl start news-perspective")
    
    def _create_startup_scripts(self):
        """Create startup scripts for different deployment scenarios"""
        
        config = self.deployments[self.deployment_type]
        
        # Production startup script
        startup_script = f"""#!/bin/bash
# Production startup script for News Perspective Diversification System

echo "Starting News Perspective Diversification System..."

# Set environment variables
export PYTHONPATH="{self.project_root}/src:$PYTHONPATH"
export OPTIMIZATION_LEVEL="{config['optimization_level']}"

# Start the application
cd {self.project_root}

if [ "$1" = "api" ]; then
    echo "Starting API server..."
    python -m uvicorn src.api.routes:app \\
        --host 0.0.0.0 \\
        --port {config['port']} \\
        --workers {config['workers']} \\
        --log-level info
elif [ "$1" = "cli" ]; then
    echo "Starting CLI browser..."
    python scripts/speed_optimized_browser.py --interactive
elif [ "$1" = "benchmark" ]; then
    echo "Running performance benchmark..."
    python scripts/speed_optimized_browser.py --benchmark
else
    echo "Usage: $0 {{api|cli|benchmark}}"
    echo "  api        - Start API server"
    echo "  cli        - Start interactive CLI"
    echo "  benchmark  - Run performance benchmark"
    exit 1
fi
"""
        
        startup_file = self.project_root / "scripts" / "start_production.sh"
        with open(startup_file, 'w') as f:
            f.write(startup_script)
        
        # Make executable
        if hasattr(os, 'chmod'):
            os.chmod(startup_file, 0o755)
        
        self.logger.info(f"Created startup script: {startup_file}")
        
        # Docker startup script
        docker_script = f"""#!/bin/bash
# Docker container startup script

echo "Initializing News Perspective Diversification System in Docker..."

# Wait for dependencies
sleep 5

# Start the application based on environment
if [ "$DEPLOYMENT_MODE" = "api" ]; then
    exec python -m uvicorn src.api.routes:app \\
        --host 0.0.0.0 \\
        --port ${{PORT:-{config['port']}}} \\
        --workers ${{WORKERS:-{config['workers']}}}
else
    exec python scripts/speed_optimized_browser.py --interactive
fi
"""
        
        docker_startup_file = self.project_root / "scripts" / "docker_startup.sh"
        with open(docker_startup_file, 'w') as f:
            f.write(docker_script)
        
        if hasattr(os, 'chmod'):
            os.chmod(docker_startup_file, 0o755)
        
        self.logger.info(f"Created Docker startup script: {docker_startup_file}")
    
    def _validate_setup(self):
        """Validate the production setup"""
        
        self.logger.info("Validating production setup...")
        
        # Check required directories exist
        required_dirs = ["data", "logs", "config", "src"]
        for directory in required_dirs:
            dir_path = self.project_root / directory
            assert dir_path.exists(), f"Required directory missing: {directory}"
        
        # Check configuration files
        config_files = ["config/expanded_news_sources.yaml"]
        for config_file in config_files:
            file_path = self.project_root / config_file
            assert file_path.exists(), f"Required config file missing: {config_file}"
        
        # Test import of core modules
        sys.path.append(str(self.project_root / "src"))
        
        try:
            from data_collection.enhanced_news_collector import EnhancedNewsCollector
            from models.optimized_perspective_matcher import OptimizedPerspectiveMatcher
            self.logger.info("Core modules import successfully")
        except ImportError as e:
            raise Exception(f"Failed to import core modules: {e}")
        
        # Test basic functionality
        try:
            matcher = OptimizedPerspectiveMatcher(
                optimization_level=self.deployments[self.deployment_type]["optimization_level"]
            )
            stats = matcher.get_performance_statistics()
            assert isinstance(stats, dict), "Performance statistics should be dict"
            self.logger.info("Basic functionality test passed")
        except Exception as e:
            raise Exception(f"Basic functionality test failed: {e}")
        
        self.logger.info("Setup validation completed successfully!")

def main():
    """Main setup function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Setup for News Perspective System")
    parser.add_argument(
        "--deployment-type", 
        choices=["development", "standard", "high_performance"],
        default="standard",
        help="Deployment configuration type"
    )
    parser.add_argument(
        "--skip-downloads", 
        action="store_true",
        help="Skip model downloads (use existing models)"
    )
    
    args = parser.parse_args()
    
    try:
        setup = ProductionSetup(args.deployment_type)
        
        if args.skip_downloads:
            setup._download_models = lambda: print("Skipping model downloads as requested")
        
        setup.setup_environment()
        
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Deployment type: {args.deployment_type}")
        print("Next steps:")
        print("1. Copy .env.production to .env and add your API keys")
        print("2. Test the setup: ./scripts/start_production.sh benchmark")
        print("3. Start the service: ./scripts/start_production.sh api")
        print("\nFor more information, see the documentation.")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
