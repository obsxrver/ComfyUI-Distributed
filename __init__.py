import os
import sys

# Add the directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import everything needed from the main module
from .multigpu import (
    NODE_CLASS_MAPPINGS, 
    NODE_DISPLAY_NAME_MAPPINGS, 
    ensure_config_exists,
    CONFIG_FILE
)

WEB_DIRECTORY = "./web"

# Use the centralized config management from multigpu.py
ensure_config_exists()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("[MultiGPU] Loaded Multi-GPU nodes.")
print(f"[MultiGPU] Config file: {CONFIG_FILE}")