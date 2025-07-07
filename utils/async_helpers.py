"""
Async helper utilities for ComfyUI-Distributed.
"""
import asyncio
import threading
from typing import Optional, Any, Coroutine
from .network import get_server_loop

def run_async_in_server_loop(coro: Coroutine, timeout: Optional[float] = None) -> Any:
    """
    Run async coroutine in server's event loop and wait for result.
    
    This is useful when you need to run async code from a synchronous context
    but want to use the server's existing event loop instead of creating a new one.
    
    Args:
        coro: The coroutine to run
        timeout: Optional timeout in seconds
        
    Returns:
        The result of the coroutine
        
    Raises:
        TimeoutError: If the operation times out
        Exception: Any exception raised by the coroutine
    """
    event = threading.Event()
    result = None
    error = None
    
    async def wrapper():
        nonlocal result, error
        try:
            result = await coro
        except Exception as e:
            error = e
        finally:
            event.set()
    
    # Schedule on server's event loop
    loop = get_server_loop()
    asyncio.run_coroutine_threadsafe(wrapper(), loop)
    
    # Wait for completion
    if not event.wait(timeout):
        raise TimeoutError(f"Async operation timed out after {timeout} seconds")
    
    if error:
        raise error
    return result