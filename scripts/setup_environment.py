# Environment setup script
# scripts/setup_environment.py

import os
from pathlib import Path
import subprocess
import sys

def setup_environment():
    """Set up the development environment"""
    
    print("Setting up News Perspective Diversification System...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    
    project_root = Path(__file__).parent.parent
    
    # Create virtual environment if it doesn't exist
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)])
    
    # Install PyTorch CPU version first
    print("Installing PyTorch (CPU version)...")
    pip_path = venv_path / "bin" / "pip" if os.name != 'nt' else venv_path / "Scripts" / "pip.exe"
    subprocess.run([
        str(pip_path), "install", "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    
    # Install remaining requirements
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        print("Installing remaining requirements...")
        subprocess.run([str(pip_path), "install", "-r", str(requirements_file)])
    
    # Create .env file template
    env_file = project_root / ".env"
    if not env_file.exists():
        print("Creating .env template...")
        with open(env_file, 'w') as f:
            f.write("""# News API Keys
NEWSAPI_API_KEY=b5a41869b3d84353b115af0d275fce44
GUARDIAN_API_KEY=ec3ce79f-450d-467b-b46e-43c097f057ca
NYTIMES_API_KEY = NBYlBcfzDotwo3Zc7wLLBIW24mgx8Mue

# Optional: Other API keys
TWITTER_API_KEY=your_twitter_key_here
""")
    
    print("\nSetup complete!")
    print(f"Virtual environment created at: {venv_path}")
    print(f"Please update {env_file} with your API keys")
    
    return True

if __name__ == "__main__":
    setup_environment()

