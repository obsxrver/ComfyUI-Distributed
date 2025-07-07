"""
Network and API utilities for ComfyUI-Distributed.
"""
import aiohttp
from aiohttp import web
from .logging import debug_log

# Shared session for connection pooling
_client_session = None

async def get_client_session():
    """Get or create a shared aiohttp client session."""
    global _client_session
    if _client_session is None or _client_session.closed:
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        # Don't set timeout here - set it per request
        _client_session = aiohttp.ClientSession(connector=connector)
    return _client_session

async def cleanup_client_session():
    """Clean up the shared client session."""
    global _client_session
    if _client_session and not _client_session.closed:
        await _client_session.close()
        _client_session = None

async def handle_api_error(request, error, status=500):
    """Standardized error response handler."""
    debug_log(f"API Error: {error}")
    return web.json_response({"status": "error", "message": str(error)}, status=status)

def get_server_port():
    """Get the ComfyUI server port."""
    import server
    return server.PromptServer.instance.port

def get_server_loop():
    """Get the ComfyUI server event loop."""
    import server
    return server.PromptServer.instance.loop

