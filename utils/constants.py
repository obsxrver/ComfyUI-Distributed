"""
Shared constants for ComfyUI-Distributed.
"""

# Timeouts (in seconds)
WORKER_JOB_TIMEOUT = 30.0
TILE_COLLECTION_TIMEOUT = 30.0
TILE_WAIT_TIMEOUT = 30.0
PROCESS_TERMINATION_TIMEOUT = 5.0

# Process monitoring
WORKER_CHECK_INTERVAL = 2.0
STATUS_CHECK_INTERVAL = 5.0

# Network
CHUNK_SIZE = 8192
LOG_TAIL_BYTES = 65536  # 64KB

# File paths
WORKER_LOG_PATTERN = "distributed_worker_*.log"

# Worker management
WORKER_STARTUP_DELAY = 2.0

# Tile transfer
TILE_TRANSFER_TIMEOUT = 30.0

# Process cleanup
PROCESS_WAIT_TIMEOUT = 3.0
QUEUE_INIT_TIMEOUT = 5.0
TILE_SEND_TIMEOUT = 60.0

# Memory operations  
MEMORY_CLEAR_DELAY = 0.5