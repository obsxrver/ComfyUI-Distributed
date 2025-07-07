"""
Process management utilities for ComfyUI-Distributed.
"""
import os
import subprocess
import platform
import signal

def is_process_alive(pid):
    """Check if a process with given PID is still alive."""
    try:
        if platform.system() == "Windows":
            # Windows: use tasklist
            result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                  capture_output=True, text=True)
            return str(pid) in result.stdout
        else:
            # Unix: send signal 0
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.SubprocessError):
        return False

def terminate_process(process, timeout=5):
    """Gracefully terminate a process with timeout."""
    if process.poll() is None:  # Still running
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

def get_python_executable():
    """Get the Python executable path."""
    import sys
    return sys.executable

