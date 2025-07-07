"""
Shared logging utilities for ComfyUI-Distributed.
"""
import os
import json

# Config file is in parent directory
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gpu_config.json")

def is_debug_enabled():
    """Check if debug is enabled."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get("settings", {}).get("debug", False)
        except:
            pass
    return False

def debug_log(message):
    """Log debug messages only if debug is enabled in config."""
    if is_debug_enabled():
        print(f"[Distributed] {message}")

def log(message):
    """Always log important messages."""
    print(f"[Distributed] {message}")