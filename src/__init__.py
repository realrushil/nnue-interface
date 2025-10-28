"""
NNUE Interface - Python bindings for Stockfish NNUE evaluation
"""

import os
import sys
from pathlib import Path
import urllib.request
import hashlib

__version__ = "0.1.0"

# NNUE file URLs and hashes
NNUE_FILES = {
    "nn-1c0000000000.nnue": {
        "url": "https://tests.stockfishchess.org/api/nn/nn-1c0000000000.nnue",
        "sha256": None,  # Will skip verification if None
    },
    "nn-37f18f62d772.nnue": {
        "url": "https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue",
        "sha256": None,
    },
}

def get_nnue_dir():
    """Get the directory where NNUE files are stored."""
    # First check if we're in development (src directory exists with .nnue files)
    src_dir = Path(__file__).parent
    if (src_dir / "nn-1c0000000000.nnue").exists():
        return src_dir
    
    # Otherwise use user cache directory
    if sys.platform == "win32":
        cache_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        cache_dir = Path.home() / "Library" / "Caches"
    else:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    
    nnue_dir = cache_dir / "stockfish_nnue"
    nnue_dir.mkdir(parents=True, exist_ok=True)
    return nnue_dir

def download_nnue_files():
    """Download NNUE files if they don't exist."""
    nnue_dir = get_nnue_dir()
    
    for filename, info in NNUE_FILES.items():
        filepath = nnue_dir / filename
        
        if filepath.exists():
            continue
        
        print(f"Downloading {filename}...", file=sys.stderr)
        try:
            urllib.request.urlretrieve(info["url"], filepath)
            
            # Verify hash if provided
            if info["sha256"]:
                with open(filepath, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                if file_hash != info["sha256"]:
                    filepath.unlink()
                    raise ValueError(f"Hash mismatch for {filename}")
            
            print(f"Successfully downloaded {filename}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to download {filename}: {e}", file=sys.stderr)
            if filepath.exists():
                filepath.unlink()

# Download files on import
download_nnue_files()

# Set environment variable so C++ code knows where to find files
os.environ["NNUE_DIR"] = str(get_nnue_dir())

# Import the C++ extension
try:
    from . import stockfish_nnue as _nnue
    
    # Re-export functions
    get_activations_and_eval = _nnue.get_activations_and_eval
    get_evaluation = _nnue.get_evaluation
    get_network_info = _nnue.get_network_info
    
    __all__ = ['get_activations_and_eval', 'get_evaluation', 'get_network_info', '__version__']
except ImportError as e:
    print(f"Warning: Failed to import stockfish_nnue C++ extension: {e}", file=sys.stderr)
    raise

