"""
Environment setup utility.

This module automatically loads the .env file and sets up cache paths
when imported. Import this BEFORE importing transformers or datasets.

"""

import os
from pathlib import Path
from dotenv import load_dotenv


def setup_environment():
    # Load .env for WANDB_API_KEY and CACHE_FOLDER
    current_file = Path(__file__)    
    project_root = current_file.parent.parent.parent  # Go up 3 levels
    env_path = project_root / '.env'
    load_dotenv(env_path)
    
    # Get cache directory from .env or use default
    cache_folder = os.environ.get('CACHE_FOLDER')
    if cache_folder:
        cache_dir = Path(cache_folder)
    else:
        # Fallback to project_root/.cache
        cache_dir = project_root / '.cache'
    
    # Set cache paths programmatically
    os.environ['HF_HOME'] = str(cache_dir / 'huggingface')
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / 'huggingface' / 'transformers')
    os.environ['HF_DATASETS_CACHE'] = str(cache_dir / 'huggingface' / 'datasets')
    os.environ['TORCH_HOME'] = str(cache_dir / 'torch')
    os.environ['PIP_CACHE_DIR'] = str(cache_dir / 'pip')
    os.environ['WANDB_DIR'] = str(cache_dir / 'wandb')
    os.environ['WANDB_CACHE_DIR'] = str(cache_dir / 'wandb')


# Auto-run setup when module is imported
setup_environment()