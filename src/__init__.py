# AutoVoice Source Code Package
# GPU-accelerated voice synthesis system

__version__ = "0.1.0"
__author__ = "AutoVoice Team"

# Import core modules
from .gpu import *
from .audio import *
from .models import *
from .training import *
from .inference import *
from .web import *
from .utils import *

# CUDA kernel extension import
try:
    from .cuda_kernels import *
except ImportError:
    pass  # Extensions built during setup

# Main entry points
# Note: main.py is at project root, not in src/
# Import from parent directory
import sys
import os
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from main import initialize_system, run_app
except ImportError:
    # Fallback if main.py not accessible
    initialize_system = None
    run_app = None

# Constants
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80