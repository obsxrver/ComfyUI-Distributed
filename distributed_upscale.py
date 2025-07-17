"""
ComfyUI Distributed Upscale Module

This module implements a distributed version of Ultimate SD Upscale with multi-mode
batch handling for efficient processing of single images and video frames.

Processing Modes:
- Single GPU: Process all tiles locally when no workers are available
- Static Mode: For small batches (≤ dynamic_threshold), flattens tiles across batch
- Dynamic Mode: For large batches, assigns whole images to workers dynamically
"""

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import json
import asyncio
import aiohttp
from aiohttp import web
import io
import server
import math
import os
import time
from typing import List, Tuple

# Import ComfyUI modules
import comfy.samplers

# Import shared utilities
from .utils.logging import debug_log, log
from .utils.config import CONFIG_FILE
from .utils.image import tensor_to_pil, pil_to_tensor
from .utils.network import get_server_port, get_server_loop, get_client_session, handle_api_error
from .utils.async_helpers import run_async_in_server_loop
from .utils.constants import (
    TILE_COLLECTION_TIMEOUT, TILE_WAIT_TIMEOUT, TILE_TRANSFER_TIMEOUT,
    QUEUE_INIT_TIMEOUT, TILE_SEND_TIMEOUT
)

# Make MAX_BATCH configurable
MAX_BATCH = int(os.environ.get('COMFYUI_MAX_BATCH', '20'))

# Configure maximum payload size (50MB default, configurable via environment variable)
MAX_PAYLOAD_SIZE = int(os.environ.get('COMFYUI_MAX_PAYLOAD_SIZE', str(50 * 1024 * 1024)))

# Helper function to ensure persistent state is initialized
def ensure_tile_jobs_initialized():
    """Ensure tile job storage is initialized on the server instance."""
    prompt_server = server.PromptServer.instance
    if not hasattr(prompt_server, 'distributed_pending_tile_jobs'):
        debug_log("Initializing persistent tile job queue on server instance.")
        prompt_server.distributed_pending_tile_jobs = {}
        prompt_server.distributed_tile_jobs_lock = asyncio.Lock()
    else:
        # Clean up any legacy queue structures that don't have the 'mode' field
        # (Should be rare after fixes, but keep for safety)
        to_remove = [job_id for job_id, job_data in prompt_server.distributed_pending_tile_jobs.items()
                     if not isinstance(job_data, dict) or 'mode' not in job_data]
        for job_id in to_remove:
            debug_log(f"Removing legacy queue structure for job {job_id}")
            del prompt_server.distributed_pending_tile_jobs[job_id]
    return prompt_server

# Note: tensor_to_pil and pil_to_tensor are imported from utils.image

class UltimateSDUpscaleDistributed:
    """
    Distributed version of Ultimate SD Upscale (No Upscale).
    
    Supports three processing modes:
    1. Single GPU: No workers available, process everything locally
    2. Static Mode: Small batches, distributes tiles across workers (flattened)
    3. Dynamic Mode: Large batches, assigns whole images to workers dynamically
    
    Features:
    - Multi-mode batch handling for efficient video/image upscaling
    - Tiled VAE support for memory efficiency
    - Dynamic load balancing for large batches
    - Backward compatible with single-image workflows
    
    Environment Variables:
    - COMFYUI_MAX_BATCH: Chunk size for tile sending (default 20)
    - COMFYUI_MAX_PAYLOAD_SIZE: Max API payload bytes (default 50MB)
    
    Threshold: dynamic_threshold input controls mode switch (default 8)
    """
    
    def __init__(self):
        """Initialize the node and ensure persistent storage exists."""
        # Pre-initialize the persistent storage on node creation
        ensure_tile_jobs_initialized()
        debug_log("UltimateSDUpscaleDistributed - Node initialized")
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscaled_image": ("IMAGE",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 256}),
                "force_uniform_tiles": ("BOOLEAN", {"default": True}),
                "tiled_decode": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "worker_id": ("STRING", {"default": ""}),
                "tile_indices": ("STRING", {"default": ""}),  # Unused - kept for compatibility
                "dynamic_threshold": ("INT", {"default": 8, "min": 1, "max": 64}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/upscaling"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution."""
        return float("nan")  # Always re-execute
    
    def run(self, upscaled_image, model, positive, negative, vae, seed, steps, cfg, 
            sampler_name, scheduler, denoise, tile_width, tile_height, padding, 
            mask_blur, force_uniform_tiles, tiled_decode, multi_job_id="", is_worker=False, 
            master_url="", enabled_worker_ids="[]", worker_id="", tile_indices="", dynamic_threshold=8):
        """Entry point - runs SYNCHRONOUSLY like Ultimate SD Upscaler."""
        if not multi_job_id:
            # No distributed processing, run single GPU version
            return self.process_single_gpu(upscaled_image, model, positive, negative, vae,
                                          seed, steps, cfg, sampler_name, scheduler, denoise,
                                          tile_width, tile_height, padding, mask_blur, force_uniform_tiles, tiled_decode)
        
        if is_worker:
            # Worker mode: process tiles synchronously
            return self.process_worker_tiles(upscaled_image, model, positive, negative, vae,
                                           seed, steps, cfg, sampler_name, scheduler, denoise,
                                           tile_width, tile_height, padding, mask_blur,
                                           force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                                           worker_id, enabled_worker_ids, dynamic_threshold)
        else:
            # Master mode: distribute and collect synchronously
            return self.process_master(upscaled_image, model, positive, negative, vae,
                                     seed, steps, cfg, sampler_name, scheduler, denoise,
                                     tile_width, tile_height, padding, mask_blur,
                                     force_uniform_tiles, tiled_decode, multi_job_id, enabled_worker_ids, dynamic_threshold)
    
    def process_worker_tiles(self, upscaled_image, model, positive, negative, vae,
                           seed, steps, cfg, sampler_name, scheduler, denoise,
                           tile_width, tile_height, padding, mask_blur,
                           force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                           worker_id, enabled_worker_ids, dynamic_threshold):
        """Worker processing - determines mode and processes accordingly."""
        # Get batch size to determine mode
        batch_size = upscaled_image.shape[0]
        
        # Ensure mode consistency across master/workers via shared threshold
        # Determine mode (must match master's logic)
        enabled_workers = json.loads(enabled_worker_ids)
        num_workers = len(enabled_workers)
        
        if batch_size == 1 or batch_size <= dynamic_threshold:
            mode = "static"
        else:
            mode = "dynamic"
            
        debug_log(f"UltimateSDUpscale Worker - Mode: {mode}, batch_size: {batch_size}")
        
        if mode == "dynamic":
            return self.process_worker_dynamic(upscaled_image, model, positive, negative, vae,
                                             seed, steps, cfg, sampler_name, scheduler, denoise,
                                             tile_width, tile_height, padding, mask_blur,
                                             force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                                             worker_id, enabled_worker_ids, dynamic_threshold)
        
        # Static mode - continue with existing tile-based processing
        # Round tile dimensions
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)
        
        # Calculate all tile positions
        _, height, width, _ = upscaled_image.shape
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        
        # Calculate total tiles across all images
        num_tiles_per_image = len(all_tiles)
        total_tiles = batch_size * num_tiles_per_image
        
        # Get assigned global tile indices for this worker
        assigned_global_indices = self._get_worker_global_indices(total_tiles, enabled_worker_ids, worker_id)
        if not assigned_global_indices:
            return (upscaled_image,)
        
        debug_log(f"UltimateSDUpscale Worker - Worker {worker_id} processing {len(assigned_global_indices)} tiles from batch of {batch_size}")
        
        # Process tiles SYNCHRONOUSLY
        processed_tiles = []
        for global_idx in assigned_global_indices:
            # Calculate which image and tile this corresponds to
            batch_idx = global_idx // num_tiles_per_image
            tile_idx = global_idx % num_tiles_per_image
            
            # Skip if batch_idx is out of range
            if batch_idx >= upscaled_image.shape[0]:
                debug_log(f"Warning: Calculated batch_idx {batch_idx} exceeds batch size {upscaled_image.shape[0]}")
                continue
                
            x, y = all_tiles[tile_idx]
            
            # Extract tile from the specific image in the batch
            tile_tensor, x1, y1, ew, eh = self.extract_tile_with_padding(
                upscaled_image[batch_idx:batch_idx+1], x, y, tile_width, tile_height, padding
            )
            
            # Process tile through SD with unique seed
            image_seed = seed + batch_idx * 1000
            processed_tile = self.process_tile(tile_tensor, model, positive, negative, vae,
                                             image_seed + tile_idx, steps, cfg, sampler_name, 
                                             scheduler, denoise, tiled_decode)
            
            processed_tiles.append({
                'tile': processed_tile,
                'global_idx': global_idx,
                'batch_idx': batch_idx,
                'tile_idx': tile_idx,
                'x': x1,  # Use extraction position
                'y': y1,
                'extracted_width': ew,
                'extracted_height': eh
            })
        
        # Send results back using the server's event loop in a single batch
        try:
            run_async_in_server_loop(
                self.send_tiles_batch_to_master(processed_tiles, multi_job_id, master_url,
                                               padding, worker_id),
                timeout=TILE_SEND_TIMEOUT
            )
        except Exception as e:
            log(f"UltimateSDUpscale Worker - Error sending tiles: {e}")
        
        return (upscaled_image,)
    
    def process_master(self, upscaled_image, model, positive, negative, vae,
                      seed, steps, cfg, sampler_name, scheduler, denoise,
                      tile_width, tile_height, padding, mask_blur,
                      force_uniform_tiles, tiled_decode, multi_job_id, enabled_worker_ids, dynamic_threshold):
        """Master processing - SYNCHRONOUS with async collection."""
        # Round tile dimensions
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)
        
        # Get image dimensions and batch size
        batch_size, height, width, _ = upscaled_image.shape
        
        # Calculate all tiles
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        total_tiles = len(all_tiles)
        
        debug_log(f"UltimateSDUpscale Master - Total tiles: {total_tiles}")
        
        # Parse enabled workers
        enabled_workers = json.loads(enabled_worker_ids)
        num_workers = len(enabled_workers)
        
        # Determine processing mode
        if num_workers == 0:
            mode = "single_gpu"
        elif batch_size == 1 or batch_size <= dynamic_threshold:
            mode = "static"
        else:
            mode = "dynamic"
        
        debug_log(f"UltimateSDUpscale Master - Mode: {mode}, batch_size: {batch_size}, workers: {num_workers}")
        
        if mode == "single_gpu":
            # No workers, process all tiles locally
            return self.process_single_gpu(upscaled_image, model, positive, negative, vae,
                                         seed, steps, cfg, sampler_name, scheduler, denoise,
                                         tile_width, tile_height, padding, mask_blur, force_uniform_tiles, tiled_decode)
        
        elif mode == "dynamic":
            # Dynamic mode for large batches
            return self.process_master_dynamic(upscaled_image, model, positive, negative, vae,
                                             seed, steps, cfg, sampler_name, scheduler, denoise,
                                             tile_width, tile_height, padding, mask_blur,
                                             force_uniform_tiles, tiled_decode, multi_job_id, enabled_workers)
        
        # Static mode - continue with existing implementation
        
        # Initialize job queue FIRST before any processing
        # Initialize the queue directly in the persistent server storage
        # This ensures it's accessible from the web server's event loop
        try:
            run_async_in_server_loop(
                self._init_job_queue(multi_job_id),
                timeout=QUEUE_INIT_TIMEOUT
            )
            debug_log(f"UltimateSDUpscale Master - Queue initialization complete")
        except Exception as e:
            debug_log(f"UltimateSDUpscale Master - Queue initialization error: {e}")
            raise RuntimeError(f"Failed to initialize static mode queue: {e}")
        
        # Don't try to prepare via API since we're using local queues
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        
        # Convert batch to PIL list for processing
        result_images = []
        for b in range(batch_size):
            image_pil = tensor_to_pil(upscaled_image[b:b+1], 0)
            result_images.append(image_pil.copy())
        
        # Mask will be created per tile for proper blending
        
        # Calculate total tiles across all images
        num_tiles_per_image = len(all_tiles)
        total_tiles_all_images = batch_size * num_tiles_per_image
        
        # Calculate master's global tile indices 
        master_global_indices = self._get_master_global_indices(total_tiles_all_images, num_workers)
        master_tiles = len(master_global_indices)
        
        debug_log(f"UltimateSDUpscale Master - Processing {master_tiles} tiles locally from batch of {batch_size}")
        
        # Process master tiles SYNCHRONOUSLY
        for global_idx in master_global_indices:
            # Calculate which image and tile this corresponds to
            batch_idx = global_idx // num_tiles_per_image
            tile_idx = global_idx % num_tiles_per_image
            
            # Skip if batch_idx is out of range
            if batch_idx >= len(result_images):
                debug_log(f"Warning: Calculated batch_idx {batch_idx} exceeds result_images length {len(result_images)}")
                continue
                
            # Use unique seed per image
            image_seed = seed + batch_idx * 1000
            
            result_images[batch_idx] = self._process_and_blend_tile(
                tile_idx, all_tiles[tile_idx], upscaled_image[batch_idx:batch_idx+1], result_images[batch_idx],
                model, positive, negative, vae, image_seed, steps, cfg,
                sampler_name, scheduler, denoise, tile_width, tile_height,
                padding, mask_blur, width, height, tiled_decode
            )
        
        # Collect worker tiles using async
        worker_tiles_expected = total_tiles_all_images - master_tiles
        if worker_tiles_expected > 0:
            # Calculate tile distribution for active worker calculation
            tiles_per_participant = total_tiles_all_images // (num_workers + 1)
            remainder = total_tiles_all_images % (num_workers + 1)
            
            # Calculate which workers will actually have tiles
            active_workers = []
            for i in range(num_workers):
                worker_start_idx = master_tiles + (i * tiles_per_participant)
                if i < remainder - 1:
                    worker_start_idx += i
                    worker_tile_count = tiles_per_participant + 1
                else:
                    worker_start_idx += remainder - 1 if remainder > 0 else 0
                    worker_tile_count = tiles_per_participant
                
                if worker_start_idx < total_tiles_all_images and worker_tile_count > 0:
                    active_workers.append(enabled_workers[i])
            
            debug_log(f"UltimateSDUpscale Master - Waiting for {worker_tiles_expected} tiles from {len(active_workers)} active workers")
            debug_log(f"UltimateSDUpscale Master - Active worker IDs: {active_workers}")
            debug_log(f"UltimateSDUpscale Master - All enabled worker IDs: {enabled_workers}")
            
            # Collect tiles from workers using the server's event loop
            collected_tiles = run_async_in_server_loop(
                self._async_collect_worker_tiles(multi_job_id, len(active_workers)),
                timeout=TILE_COLLECTION_TIMEOUT
            )
            
            # Blend collected tiles SYNCHRONOUSLY
            for tile_data in collected_tiles.values():
                    x = tile_data['x']
                    y = tile_data['y']
                    tile_tensor = tile_data['tensor']
                    tile_idx = tile_data['tile_idx']
                    extracted_width = tile_data.get('extracted_width', tile_width + 2 * padding)
                    extracted_height = tile_data.get('extracted_height', tile_height + 2 * padding)
                    
                    # Get batch index if available (for batch processing)
                    batch_idx = tile_data.get('batch_idx', 0)
                    
                    # Skip if batch_idx is out of range
                    if batch_idx >= len(result_images):
                        debug_log(f"Warning: Received tile for batch_idx {batch_idx} but only have {len(result_images)} images")
                        continue
                    
                    # Convert and blend
                    tile_pil = tensor_to_pil(tile_tensor, 0)
                    # Get the original tile position from the tile index
                    orig_x, orig_y = all_tiles[tile_idx]
                    # Create mask for this specific tile
                    tile_mask = self.create_tile_mask(width, height, orig_x, orig_y, tile_width, tile_height, mask_blur)
                    # Use extraction position and size for blending
                    result_images[batch_idx] = self.blend_tile(result_images[batch_idx], tile_pil, 
                                                 x, y, (extracted_width, extracted_height), tile_mask, padding)
        
        # Convert back to tensor
        # For batch processing, combine all result images
        if batch_size == 1:
            result_tensor = pil_to_tensor(result_images[0])
        else:
            # Convert all images and concatenate
            result_tensors = [pil_to_tensor(img) for img in result_images]
            result_tensor = torch.cat(result_tensors, dim=0)
        
        if upscaled_image.is_cuda:
            result_tensor = result_tensor.cuda()
        
        debug_log(f"UltimateSDUpscale Master - Job {multi_job_id} complete")
        return (result_tensor,)
    
    def _get_worker_global_indices(self, total_tiles, enabled_worker_ids, worker_id):
        """Calculate which global tile indices are assigned to a specific worker in flattened mode."""
        enabled_workers = json.loads(enabled_worker_ids)
        num_workers = len(enabled_workers)
        
        debug_log(f"UltimateSDUpscale Worker - Worker ID: {worker_id}, Enabled workers: {enabled_workers}")
        
        try:
            worker_index = enabled_workers.index(worker_id)
        except ValueError:
            log(f"UltimateSDUpscale Worker - Worker {worker_id} not found in enabled list {enabled_workers}")
            return []
        
        # Calculate tile distribution
        tiles_per_participant = total_tiles // (num_workers + 1)
        remainder = total_tiles % (num_workers + 1)
        master_tiles = tiles_per_participant + (1 if remainder > 0 else 0)
        
        start_idx = master_tiles + (worker_index * tiles_per_participant)
        if worker_index < remainder - 1:
            start_idx += worker_index
            end_idx = start_idx + tiles_per_participant + 1
        else:
            start_idx += remainder - 1 if remainder > 0 else 0
            end_idx = start_idx + tiles_per_participant
        
        end_idx = min(end_idx, total_tiles)
        return list(range(start_idx, end_idx))
    
    def _get_worker_tiles(self, all_tiles, enabled_worker_ids, worker_id):
        """Calculate which tiles are assigned to a specific worker."""
        enabled_workers = json.loads(enabled_worker_ids)
        num_workers = len(enabled_workers)
        total_tiles = len(all_tiles)
        
        debug_log(f"UltimateSDUpscale Worker - Worker ID: {worker_id}, Enabled workers: {enabled_workers}")
        
        try:
            worker_index = enabled_workers.index(worker_id)
        except ValueError:
            log(f"UltimateSDUpscale Worker - Worker {worker_id} not found in enabled list {enabled_workers}")
            return []
        
        # Calculate tile distribution
        tiles_per_participant = total_tiles // (num_workers + 1)
        remainder = total_tiles % (num_workers + 1)
        master_tiles = tiles_per_participant + (1 if remainder > 0 else 0)
        
        start_idx = master_tiles + (worker_index * tiles_per_participant)
        if worker_index < remainder - 1:
            start_idx += worker_index
            end_idx = start_idx + tiles_per_participant + 1
        else:
            start_idx += remainder - 1 if remainder > 0 else 0
            end_idx = start_idx + tiles_per_participant
        
        end_idx = min(end_idx, total_tiles)
        return list(range(start_idx, end_idx))
    
    def _get_master_global_indices(self, total_tiles, num_workers):
        """Calculate which global tile indices are assigned to the master in flattened mode."""
        tiles_per_participant = total_tiles // (num_workers + 1)
        remainder = total_tiles % (num_workers + 1)
        master_tiles = tiles_per_participant + (1 if remainder > 0 else 0)
        return list(range(master_tiles))
    
    def _get_master_tiles(self, all_tiles, num_workers):
        """Calculate which tiles are assigned to the master."""
        total_tiles = len(all_tiles)
        tiles_per_participant = total_tiles // (num_workers + 1)
        remainder = total_tiles % (num_workers + 1)
        master_tiles = tiles_per_participant + (1 if remainder > 0 else 0)
        return list(range(master_tiles))
    
    def _process_and_blend_tile(self, tile_idx, tile_pos, upscaled_image, result_image,
                               model, positive, negative, vae, seed, steps, cfg,
                               sampler_name, scheduler, denoise, tile_width, tile_height,
                               padding, mask_blur, image_width, image_height, tiled_decode):
        """Process a single tile and blend it into the result image."""
        x, y = tile_pos
        
        # Extract and process tile
        tile_tensor, x1, y1, ew, eh = self.extract_tile_with_padding(
            upscaled_image, x, y, tile_width, tile_height, padding
        )
        
        processed_tile = self.process_tile(tile_tensor, model, positive, negative, vae,
                                         seed + tile_idx, steps, cfg, sampler_name, 
                                         scheduler, denoise, tiled_decode)
        
        # Convert and blend
        processed_pil = tensor_to_pil(processed_tile, 0)
        # Create mask for this specific tile
        tile_mask = self.create_tile_mask(image_width, image_height, x, y, tile_width, tile_height, mask_blur)
        # Use extraction position and size for blending
        result_image = self.blend_tile(result_image, processed_pil, 
                                     x1, y1, (ew, eh), tile_mask, padding)
        
        return result_image
    
    async def _prepare_multigpu_job(self, multi_job_id):
        """Prepare job via API endpoint to ensure server is ready."""
        session = await get_client_session()
        # Use the actual server port (get from server instance)
        port = get_server_port()
        async with session.post(f"http://localhost:{port}/distributed/prepare_job",
                              json={"multi_job_id": multi_job_id}) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Failed to prepare job: {response.status} - {text}")
    
    async def _init_job_queue_dynamic(self, multi_job_id, batch_size, assigned_to_workers=None, worker_status=None, worker_indices=None):
        """Initialize the job queue for dynamic mode with pending images."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_tile_jobs:
                prompt_server.distributed_pending_tile_jobs[multi_job_id] = {
                    'queue': asyncio.Queue(),
                    'mode': 'dynamic',
                    'pending_images': asyncio.Queue(),
                    'completed_images': {},
                    'completed_tiles': {},  # For uniformity
                    'batch_size': batch_size,
                    'assigned_to_workers': assigned_to_workers or {},
                    'worker_status': worker_status or {}
                }
                # Initialize pending images queue with worker indices only
                pending_images = prompt_server.distributed_pending_tile_jobs[multi_job_id]['pending_images']
                for i in worker_indices:
                    await pending_images.put(i)
                debug_log(f"UltimateSDUpscale Master Dynamic - Initialized job queue with {len(worker_indices)} pending images for workers")
            else:
                debug_log(f"UltimateSDUpscale Master Dynamic - Queue already exists for {multi_job_id}")
    
    async def _init_job_queue(self, multi_job_id):
        """Initialize the job queue for collecting tiles."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            debug_log(f"UltimateSDUpscale Master - _init_job_queue: Checking if {multi_job_id} exists in distributed_pending_tile_jobs")
            debug_log(f"UltimateSDUpscale Master - _init_job_queue: Current jobs: {list(prompt_server.distributed_pending_tile_jobs.keys())}")
            if multi_job_id not in prompt_server.distributed_pending_tile_jobs:
                prompt_server.distributed_pending_tile_jobs[multi_job_id] = {
                    'queue': asyncio.Queue(),
                    'mode': 'static',
                    'completed_tiles': {}  # For static mode data
                }
                debug_log(f"UltimateSDUpscale Master - Initialized job queue for {multi_job_id}")
                debug_log(f"UltimateSDUpscale Master - _init_job_queue: Created new queue for {multi_job_id}")
            else:
                debug_log(f"UltimateSDUpscale Master - _init_job_queue: Queue already exists for {multi_job_id}")
    
    async def _async_collect_worker_tiles(self, multi_job_id, num_workers):
        """Async helper to collect tiles from workers."""
        # Get the already initialized queue
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_tile_jobs:
                raise RuntimeError(f"Job queue not initialized for {multi_job_id}")
            job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
            if not isinstance(job_data, dict) or 'mode' not in job_data:
                raise RuntimeError("Invalid job data structure")
            if job_data['mode'] != 'static':
                raise RuntimeError(f"Mode mismatch: expected static, got {job_data['mode']}")
            q = job_data['queue']
        
        collected_tiles = {}
        workers_done = set()
        timeout = TILE_WAIT_TIMEOUT
        
        debug_log(f"UltimateSDUpscale Master - Starting collection, expecting {num_workers} workers to complete")
        
        while len(workers_done) < num_workers:
            debug_log(f"UltimateSDUpscale Master - Loop status: {len(workers_done)}/{num_workers} workers done, {len(collected_tiles)} tiles collected, workers_done set: {workers_done}")
            try:
                result = await asyncio.wait_for(q.get(), timeout=timeout)
                worker_id = result['worker_id']
                is_last = result.get('is_last', False)
                
                # Check if batch mode
                tiles = result.get('tiles', [])
                if tiles:
                    # Batch mode
                    debug_log(f"UltimateSDUpscale Master - Received batch of {len(tiles)} tiles from worker '{worker_id}' (is_last={is_last})")
                    
                    for tile_data in tiles:
                        # Validate required fields
                        if 'batch_idx' not in tile_data:
                            log(f"UltimateSDUpscale Master - Missing batch_idx in tile data, skipping")
                            continue
                        batch_idx = tile_data['batch_idx']
                        
                        tile_idx = tile_data['tile_idx']
                        # Use global_idx as key if available (for batch processing)
                        key = tile_data.get('global_idx', tile_idx)
                        
                        # Store the full tile data including metadata
                        collected_tiles[key] = {
                            'tensor': tile_data['tensor'],
                            'tile_idx': tile_idx,
                            'x': tile_data['x'],
                            'y': tile_data['y'],
                            'extracted_width': tile_data['extracted_width'],
                            'extracted_height': tile_data['extracted_height'],
                            'padding': tile_data['padding'],
                            'worker_id': worker_id,
                            'batch_idx': tile_data.get('batch_idx', 0),
                            'global_idx': tile_data.get('global_idx', tile_idx)
                        }
                else:
                    # Single tile mode (backward compat)
                    tile_idx = result['tile_idx']
                    collected_tiles[tile_idx] = result
                    debug_log(f"UltimateSDUpscale Master - Received single tile {tile_idx} from worker '{worker_id}' (is_last={is_last})")
                
                if is_last:
                    workers_done.add(worker_id)
                    debug_log(f"UltimateSDUpscale Master - Worker {worker_id} completed")
                    debug_log(f"UltimateSDUpscale Master - Worker '{worker_id}' marked as done. Total workers done: {len(workers_done)}/{num_workers}")
                
            except asyncio.TimeoutError:
                log(f"UltimateSDUpscale Master - Timeout waiting for tiles")
                debug_log(f"UltimateSDUpscale Master - Final status: {len(workers_done)}/{num_workers} workers done, {len(collected_tiles)} tiles collected")
                debug_log(f"UltimateSDUpscale Master - Workers that completed: {workers_done}")
                break
        
        debug_log(f"UltimateSDUpscale Master - Collection complete. Got {len(collected_tiles)} tiles from {len(workers_done)} workers")
        
        # Clean up job queue
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                del prompt_server.distributed_pending_tile_jobs[multi_job_id]
        
        return collected_tiles
    
    async def _async_process_dynamic(self, multi_job_id, expected_from_workers, num_workers):
        """Async helper for dynamic mode - collect processed whole images from workers.
        
        This method implements dynamic load balancing where workers request images
        as they become available, ensuring optimal utilization of GPU resources."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_tile_jobs:
                raise RuntimeError(f"Job queue not initialized for {multi_job_id}")
            job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
            if not isinstance(job_data, dict) or 'mode' not in job_data:
                raise RuntimeError("Invalid job data structure")
            if job_data['mode'] != 'dynamic':
                raise RuntimeError(f"Mode mismatch: expected dynamic, got {job_data['mode']}")
            q = job_data['queue']
            completed_images = job_data['completed_images']
        
        workers_done = set()
        timeout = TILE_WAIT_TIMEOUT * 2  # Longer timeout for full images
        
        debug_log(f"UltimateSDUpscale Master Dynamic - Waiting for {expected_from_workers} images from {num_workers} workers")
        
        last_heartbeat_check = time.time()
        
        while len(completed_images) < expected_from_workers and len(workers_done) < num_workers:
            try:
                result = await asyncio.wait_for(q.get(), timeout=10.0)  # Shorter timeout for regular checks
                worker_id = result['worker_id']
                is_last = result.get('is_last', False)
                
                if 'image_idx' in result and 'image' in result:
                    image_idx = result['image_idx']
                    image_pil = result['image']
                    completed_images[image_idx] = image_pil
                    debug_log(f"UltimateSDUpscale Master Dynamic - Received image {image_idx} from worker {worker_id}")
                
                if is_last:
                    workers_done.add(worker_id)
                    debug_log(f"UltimateSDUpscale Master Dynamic - Worker {worker_id} completed")
                    
            except asyncio.TimeoutError:
                # Check for worker timeouts every 10 seconds
                current_time = time.time()
                if current_time - last_heartbeat_check >= 10.0:
                    async with prompt_server.distributed_tile_jobs_lock:
                        if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                            job_data_check = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                            # Check for timed out workers (60 second timeout)
                            for worker, last_heartbeat in list(job_data_check.get('worker_status', {}).items()):
                                if current_time - last_heartbeat > 60:
                                    log(f"UltimateSDUpscale Master Dynamic - Worker {worker} timed out")
                                    # Requeue assigned images from this worker
                                    for idx in job_data_check.get('assigned_to_workers', {}).get(worker, []):
                                        if idx not in completed_images:
                                            await job_data_check['pending_images'].put(idx)
                                            debug_log(f"UltimateSDUpscale Master Dynamic - Requeued image {idx} from timed out worker {worker}")
                                    # Mark worker as done and clean up
                                    workers_done.add(worker)
                                    if 'worker_status' in job_data_check:
                                        del job_data_check['worker_status'][worker]
                                    if 'assigned_to_workers' in job_data_check:
                                        job_data_check['assigned_to_workers'][worker] = []
                    last_heartbeat_check = current_time
                
                # Check if we've been waiting too long overall
                if current_time - last_heartbeat_check > timeout:
                    log(f"UltimateSDUpscale Master Dynamic - Overall timeout waiting for images")
                    # Requeue all unfinished images
                    async with prompt_server.distributed_tile_jobs_lock:
                        if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                            pending_queue = prompt_server.distributed_pending_tile_jobs[multi_job_id]['pending_images']
                            for idx in range(job_data_check.get('batch_size', 0)):
                                if idx not in completed_images:
                                    await pending_queue.put(idx)
                                    debug_log(f"UltimateSDUpscale Master Dynamic - Requeued image {idx}")
                    break
        
        debug_log(f"UltimateSDUpscale Master Dynamic - Collection complete. Got {len(completed_images)} images")
        
        # Clean up job queue
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                del prompt_server.distributed_pending_tile_jobs[multi_job_id]
        
        return completed_images
    
    def round_to_multiple(self, value: int, multiple: int = 8) -> int:
        """Round value to nearest multiple."""
        return round(value / multiple) * multiple
    
    def calculate_tiles(self, image_width: int, image_height: int, 
                       tile_width: int, tile_height: int, force_uniform_tiles: bool = True) -> List[Tuple[int, int]]:
        """Calculate tile positions for the image.
        
        Tiles are placed at grid positions without overlap in their placement.
        The overlap happens during extraction with padding.
        
        If force_uniform_tiles is True, tiles are distributed evenly across the image."""
        if force_uniform_tiles:
            # Calculate number of tiles needed
            num_tiles_x = math.ceil(image_width / tile_width)
            num_tiles_y = math.ceil(image_height / tile_height)
            
            # Calculate uniform spacing
            tiles = []
            
            # Handle X dimension
            if num_tiles_x == 1:
                tiles_x = [0]
            else:
                # Calculate step to distribute tiles evenly
                step_x = (image_width - tile_width) / (num_tiles_x - 1)
                # Check for excessive overlap
                if step_x < tile_width / 2:
                    debug_log(f"Warning: Uniform tiles would have excessive overlap (step_x={step_x}, tile_width={tile_width}). Falling back to non-uniform mode.")
                    force_uniform_tiles = False
                else:
                    tiles_x = [round(i * step_x) for i in range(num_tiles_x)]
                    # Ensure tiles are aligned to 8-pixel boundaries
                    tiles_x = [self.round_to_multiple(x, 8) for x in tiles_x]
            
            # Handle Y dimension (if still using uniform tiles)
            if force_uniform_tiles:
                if num_tiles_y == 1:
                    tiles_y = [0]
                else:
                    # Calculate step to distribute tiles evenly
                    step_y = (image_height - tile_height) / (num_tiles_y - 1)
                    # Check for excessive overlap
                    if step_y < tile_height / 2:
                        debug_log(f"Warning: Uniform tiles would have excessive overlap (step_y={step_y}, tile_height={tile_height}). Falling back to non-uniform mode.")
                        force_uniform_tiles = False
                    else:
                        tiles_y = [round(i * step_y) for i in range(num_tiles_y)]
                        # Ensure tiles are aligned to 8-pixel boundaries
                        tiles_y = [self.round_to_multiple(y, 8) for y in tiles_y]
        
        # If still using uniform tiles, generate positions
        if force_uniform_tiles:
            # Generate all tile positions
            for y in tiles_y:
                for x in tiles_x:
                    tiles.append((x, y))
                    
            return tiles
        
        # Fall through to non-uniform mode if needed
        if not force_uniform_tiles:
            # Original non-uniform tile calculation
            tiles = []
            for y in range(0, image_height, tile_height):
                for x in range(0, image_width, tile_width):
                    tiles.append((x, y))
            return tiles
    
    def extract_tile_with_padding(self, image: torch.Tensor, x: int, y: int,
                                 tile_width: int, tile_height: int, padding: int) -> Tuple[torch.Tensor, int, int, int, int]:
        """Extract a tile from the image with padding."""
        _, h, w, _ = image.shape
        
        # Calculate extraction bounds with padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + tile_width + padding)
        y2 = min(h, y + tile_height + padding)
        
        # Store the actual extracted size before resizing
        extracted_width = x2 - x1
        extracted_height = y2 - y1
        
        # Debug logging (uncomment if needed)
        # debug_log(f"[Extract] Tile at ({x}, {y}) -> Extract from ({x1}, {y1}) to ({x2}, {y2}), size: {extracted_width}x{extracted_height}")
        
        # Extract tile
        tile = image[:, y1:y2, x1:x2, :]
        
        # Convert to PIL using utility function
        tile_pil = tensor_to_pil(tile, 0)
        
        # Resize to target dimensions for processing
        tile_pil = tile_pil.resize((tile_width, tile_height), Image.LANCZOS)
        
        # Convert back to tensor using utility function
        tile_tensor = pil_to_tensor(tile_pil)
        
        # Move to same device as original image
        if image.is_cuda:
            tile_tensor = tile_tensor.cuda()
        
        return tile_tensor, x1, y1, extracted_width, extracted_height
    
    def process_tile(self, tile_tensor: torch.Tensor, model, positive, negative, vae,
                    seed: int, steps: int, cfg: float, sampler_name: str, 
                    scheduler: str, denoise: float, tiled_decode: bool = False) -> torch.Tensor:
        """Process a single tile through SD sampling."""
        # Import here to avoid circular dependencies
        from nodes import common_ksampler, VAEEncode, VAEDecode
        
        # Try to import tiled VAE nodes if available
        try:
            from nodes import VAEEncodeTiled, VAEDecodeTiled
            tiled_vae_available = True
        except ImportError:
            tiled_vae_available = False
            if tiled_decode:
                debug_log("Tiled VAE nodes not available, falling back to standard VAE")
        
        # Convert to PIL and back to ensure clean tensor without gradient tracking
        tile_pil = tensor_to_pil(tile_tensor, 0)
        clean_tensor = pil_to_tensor(tile_pil)
        
        # Move to correct device
        if tile_tensor.is_cuda:
            clean_tensor = clean_tensor.cuda()
        
        # Encode to latent
        if tiled_decode and tiled_vae_available:
            latent = VAEEncodeTiled().encode(vae, clean_tensor)[0]
        else:
            latent = VAEEncode().encode(vae, clean_tensor)[0]
        
        # Sample
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                positive, negative, latent, denoise=denoise)[0]
        
        # Decode back to image
        if tiled_decode and tiled_vae_available:
            image = VAEDecodeTiled().decode(vae, samples)[0]
        else:
            image = VAEDecode().decode(vae, samples)[0]
        
        return image
    
    def create_tile_mask(self, image_width: int, image_height: int,
                        x: int, y: int, tile_width: int, tile_height: int, 
                        mask_blur: int) -> Image.Image:
        """Create a mask for blending tiles - matches Ultimate SD Upscale approach.
        
        Creates a black image with a white rectangle at the tile position,
        then applies blur to create soft edges.
        """
        # Create a full-size mask matching the image dimensions
        mask = Image.new('L', (image_width, image_height), 0)  # Black background
        
        # Draw white rectangle at tile position
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x, y, x + tile_width, y + tile_height], fill=255)
        
        # Apply blur to soften edges
        if mask_blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(mask_blur))
        
        return mask
    
    def blend_tile(self, base_image: Image.Image, tile_image: Image.Image,
                  x: int, y: int, extracted_size: Tuple[int, int], 
                  mask: int, padding: int) -> Image.Image:
        """Blend a processed tile back into the base image using Ultimate SD Upscale's exact approach.
        
        This follows the exact method from ComfyUI_UltimateSDUpscale/modules/processing.py
        """
        extracted_width, extracted_height = extracted_size
        
        # Debug logging (uncomment if needed)
        # debug_log(f"[Blend] Placing tile at ({x}, {y}), size: {extracted_width}x{extracted_height}")
        
        # Calculate the crop region that was used for extraction
        crop_region = (x, y, x + extracted_width, y + extracted_height)
        
        # The mask is already full-size, no need to crop
        
        # Resize the processed tile back to the extracted size
        if tile_image.size != (extracted_width, extracted_height):
            tile_resized = tile_image.resize((extracted_width, extracted_height), Image.LANCZOS)
        else:
            tile_resized = tile_image
        
        # Follow Ultimate SD Upscale blending approach:
        # Put the tile into position
        image_tile_only = Image.new('RGBA', base_image.size)
        image_tile_only.paste(tile_resized, crop_region[:2])
        
        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        temp.putalpha(mask)  # Use the full image mask
        image_tile_only.paste(temp, image_tile_only)
        
        # Add back the tile to the initial image according to the mask in the alpha channel
        result = base_image.convert('RGBA')
        result.alpha_composite(image_tile_only)
        
        # Convert back to RGB
        return result.convert('RGB')
    
    
    async def send_tiles_batch_to_master(self, processed_tiles, multi_job_id, master_url, 
                                       padding, worker_id):
        """Send all processed tiles to master, chunked if large."""
        if not processed_tiles:
            return  # Early exit if empty
        
        total_tiles = len(processed_tiles)
        debug_log(f"UltimateSDUpscale Worker - Sending {total_tiles} tiles in chunks of max {MAX_BATCH}")
        
        for start in range(0, total_tiles, MAX_BATCH):
            chunk = processed_tiles[start:start + MAX_BATCH]
            chunk_size = len(chunk)
            is_chunk_last = (start + chunk_size == total_tiles)  # True only for final chunk
            
            debug_log(f"UltimateSDUpscale Worker - Processing chunk {start//MAX_BATCH + 1}, tiles {start} to {start + chunk_size - 1}")
            
            data = aiohttp.FormData()
            data.add_field('multi_job_id', multi_job_id)
            data.add_field('worker_id', str(worker_id))
            data.add_field('is_last', str(is_chunk_last))
            data.add_field('batch_size', str(chunk_size))
            data.add_field('padding', str(padding))
            
            # Chunk metadata: Keep absolute tile_idx and batch_idx if present
            metadata = []
            for tile_data in chunk:
                meta = {
                    'tile_idx': tile_data['tile_idx'],  # Original/absolute idx
                    'x': tile_data['x'],
                    'y': tile_data['y'],
                    'extracted_width': tile_data['extracted_width'],
                    'extracted_height': tile_data['extracted_height']
                }
                # Add batch_idx and global_idx if present (for batch processing)
                if 'batch_idx' in tile_data:
                    meta['batch_idx'] = tile_data['batch_idx']
                if 'global_idx' in tile_data:
                    meta['global_idx'] = tile_data['global_idx']
                metadata.append(meta)
            
            # Add JSON metadata
            data.add_field('tiles_metadata', json.dumps(metadata), content_type='application/json')
            
            # Add chunk images as 'tile_{j}' where j=0 to chunk_size-1
            for j, tile_data in enumerate(chunk):
                # Convert tensor to PIL
                img = tensor_to_pil(tile_data['tile'], 0)
                byte_io = io.BytesIO()
                img.save(byte_io, format='PNG', compress_level=0)
                byte_io.seek(0)
                
                # Add image field
                data.add_field(f'tile_{j}', byte_io, filename=f'tile_{j}.png', content_type='image/png')
            
            # Estimate payload size for logging
            payload_size = 0
            for field in data._fields:
                if hasattr(field[2], 'read'):
                    field[2].seek(0)
                    payload_size += len(field[2].read())
                    field[2].seek(0)
            debug_log(f"UltimateSDUpscale Worker - Chunk payload size: {payload_size:,} bytes")
            
            # Retry logic with exponential backoff
            max_retries = 5
            retry_delay = 0.5
            
            for attempt in range(max_retries):
                try:
                    session = await get_client_session()
                    url = f"{master_url}/distributed/tile_complete"
                    
                    debug_log(f"UltimateSDUpscale Worker - Sending chunk of {chunk_size} tiles to {url}, attempt {attempt + 1}")
                    
                    async with session.post(url, data=data) as response:
                        response.raise_for_status()
                        debug_log(f"UltimateSDUpscale Worker - Successfully sent chunk of {chunk_size} tiles")
                        break  # Success, move to next chunk
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        debug_log(f"UltimateSDUpscale Worker - Retry {attempt + 1}/{max_retries} after error: {e}")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        log(f"UltimateSDUpscale Worker - Failed to send chunk after {max_retries} attempts: {e}")
                        raise

    def process_single_gpu(self, upscaled_image, model, positive, negative, vae,
                          seed, steps, cfg, sampler_name, scheduler, denoise,
                          tile_width, tile_height, padding, mask_blur, force_uniform_tiles, tiled_decode):
        """Process all tiles on a single GPU (no distribution)."""
        # Round tile dimensions
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)
        
        # Get image dimensions and batch size
        batch_size, height, width, _ = upscaled_image.shape
        
        # Calculate all tiles
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        
        debug_log(f"UltimateSDUpscale - Processing {len(all_tiles)} tiles locally for batch of {batch_size}")
        
        # Convert batch to PIL list
        result_images = []
        for b in range(batch_size):
            image_pil = tensor_to_pil(upscaled_image[b:b+1], 0)
            # Ensure RGB mode for consistency
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            result_images.append(image_pil.copy())
        
        # Process each image in the batch
        for batch_idx in range(batch_size):
            # Process each tile for this image
            for tile_idx, tile_pos in enumerate(all_tiles):
                # Use unique seed per image
                image_seed = seed + batch_idx * 1000
                result_images[batch_idx] = self._process_and_blend_tile(
                    tile_idx, tile_pos, upscaled_image[batch_idx:batch_idx+1], result_images[batch_idx],
                    model, positive, negative, vae, image_seed, steps, cfg,
                    sampler_name, scheduler, denoise, tile_width, tile_height,
                    padding, mask_blur, width, height, tiled_decode
                )
        
        # Convert back to tensor
        if batch_size == 1:
            result_tensor = pil_to_tensor(result_images[0])
        else:
            result_tensors = [pil_to_tensor(img) for img in result_images]
            result_tensor = torch.cat(result_tensors, dim=0)
            
        if upscaled_image.is_cuda:
            result_tensor = result_tensor.cuda()
        
        return (result_tensor,)
    
    def process_master_dynamic(self, upscaled_image, model, positive, negative, vae,
                              seed, steps, cfg, sampler_name, scheduler, denoise,
                              tile_width, tile_height, padding, mask_blur,
                              force_uniform_tiles, tiled_decode, multi_job_id, enabled_workers):
        """Dynamic mode for large batches - assigns whole images to workers dynamically."""
        # Get batch size and dimensions
        batch_size, height, width, _ = upscaled_image.shape
        num_workers = len(enabled_workers)
        
        debug_log(f"UltimateSDUpscale Master Dynamic - Processing batch of {batch_size} images with {num_workers} workers")
        
        # Calculate master's share similar to static mode
        num_participants = num_workers + 1
        images_per_participant = batch_size // num_participants
        remainder = batch_size % num_participants
        master_images_count = images_per_participant + (1 if remainder > 0 else 0)
        master_indices = list(range(master_images_count))
        worker_indices = list(range(master_images_count, batch_size))
        expected_from_workers = len(worker_indices)
        
        # Calculate tiles for processing
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        
        # Initialize job queue for communication - synchronous approach
        try:
            # Add worker tracking data
            assigned_to_workers = {w: [] for w in enabled_workers}
            worker_status = {w: time.time() for w in enabled_workers}
            
            # Use short timeout for async init, with threading fallback
            try:
                run_async_in_server_loop(
                    self._init_job_queue_dynamic(multi_job_id, batch_size, assigned_to_workers, worker_status, worker_indices),
                    timeout=2.0  # Short timeout
                )
            except Exception as init_error:
                debug_log(f"UltimateSDUpscale Master Dynamic - Async init failed, using threading: {init_error}")
                # Threading fallback for dynamic mode
                import threading
                def sync_dynamic_init():
                    prompt_server = ensure_tile_jobs_initialized()
                    if multi_job_id not in prompt_server.distributed_pending_tile_jobs:
                        prompt_server.distributed_pending_tile_jobs[multi_job_id] = {
                            'queue': asyncio.Queue(),
                            'mode': 'dynamic',
                            'pending_images': asyncio.Queue(),
                            'completed_images': {},
                            'completed_tiles': {},
                            'batch_size': batch_size,
                            'assigned_to_workers': assigned_to_workers,
                            'worker_status': worker_status
                        }
                        # Initialize pending images queue with worker indices only
                        pending_images = prompt_server.distributed_pending_tile_jobs[multi_job_id]['pending_images']
                        for i in worker_indices:
                            pending_images.put_nowait(i)
                        debug_log(f"Synchronous dynamic init for {multi_job_id} with {len(worker_indices)} pending images for workers")
                thread = threading.Thread(target=sync_dynamic_init)
                thread.start()
                thread.join(timeout=3.0)
                if thread.is_alive():
                    raise RuntimeError("Dynamic queue init failed")
        except Exception as e:
            debug_log(f"UltimateSDUpscale Master Dynamic - Queue initialization error: {e}")
            raise RuntimeError(f"Failed to initialize dynamic mode queue: {e}")
        
        # Convert batch to PIL list
        result_images = []
        for b in range(batch_size):
            image_pil = tensor_to_pil(upscaled_image[b:b+1], 0)
            # Ensure RGB mode for consistency
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            result_images.append(image_pil.copy())
        
        # Process master's share locally
        debug_log(f"UltimateSDUpscale Master Dynamic - Processing {master_images_count} images locally")
        for idx in master_indices:
            single_tensor = upscaled_image[idx:idx+1]
            local_image = result_images[idx]
            image_seed = seed + idx * 1000
            for tile_idx, pos in enumerate(all_tiles):
                local_image = self._process_and_blend_tile(
                    tile_idx, pos, single_tensor, local_image,
                    model, positive, negative, vae, image_seed, steps, cfg,
                    sampler_name, scheduler, denoise, tile_width, tile_height,
                    padding, mask_blur, width, height, tiled_decode
                )
            result_images[idx] = local_image
        
        # Process with workers using dynamic assignment
        collected_images = run_async_in_server_loop(
            self._async_process_dynamic(multi_job_id, expected_from_workers, num_workers),
            timeout=TILE_COLLECTION_TIMEOUT * 2  # Longer timeout for full images
        )
        
        # Update result images with processed ones
        for idx, processed_img in collected_images.items():
            if idx < batch_size:
                result_images[idx] = processed_img
        
        # Convert back to tensor
        if batch_size == 1:
            result_tensor = pil_to_tensor(result_images[0])
        else:
            result_tensors = [pil_to_tensor(img) for img in result_images]
            result_tensor = torch.cat(result_tensors, dim=0)
        
        if upscaled_image.is_cuda:
            result_tensor = result_tensor.cuda()
        
        debug_log(f"UltimateSDUpscale Master Dynamic - Job {multi_job_id} complete")
        return (result_tensor,)
    
    def process_worker_dynamic(self, upscaled_image, model, positive, negative, vae,
                                  seed, steps, cfg, sampler_name, scheduler, denoise,
                                  tile_width, tile_height, padding, mask_blur,
                                  force_uniform_tiles, tiled_decode, multi_job_id, master_url,
                                  worker_id, enabled_worker_ids, dynamic_threshold):
            """Worker processing in dynamic mode - processes whole images."""
            # Round tile dimensions
            tile_width = self.round_to_multiple(tile_width)
            tile_height = self.round_to_multiple(tile_height)
            
            # Get dimensions
            _, height, width, _ = upscaled_image.shape
            all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
            
            debug_log(f"UltimateSDUpscale Worker Dynamic - Worker {worker_id} ready to process images")
            
            # Keep track of processed images for is_last detection
            processed_count = 0
            
            # Poll for job readiness to avoid races during master init
            max_poll_attempts = 20  # ~20s at 1s sleep
            for attempt in range(max_poll_attempts):
                ready = run_async_in_server_loop(
                    self._check_job_status(multi_job_id, master_url),
                    timeout=5.0
                )
                if ready:
                    debug_log(f"Job {multi_job_id} ready after {attempt} attempts")
                    break
                time.sleep(1.0)  # Poll every 1s
            else:
                log(f"Job {multi_job_id} not ready after {max_poll_attempts} attempts, aborting")
                return (upscaled_image,)
            
            # Loop to request and process images
            while True:
                # Request an image to process
                image_idx, estimated_remaining = run_async_in_server_loop(
                    self._request_image_from_master(multi_job_id, master_url, worker_id),
                    timeout=TILE_WAIT_TIMEOUT
                )
                
                if image_idx is None:
                    debug_log(f"UltimateSDUpscale Worker Dynamic - No more images to process")
                    break
                    
                debug_log(f"UltimateSDUpscale Worker Dynamic - Mode: dynamic, assigned image {image_idx} to worker {worker_id}")
                processed_count += 1
                
                # Determine if this should be marked as last for this worker
                is_last_for_worker = (estimated_remaining == 0)
                
                # Extract single image tensor
                single_tensor = upscaled_image[image_idx:image_idx+1]
                
                # Convert to PIL for processing
                local_image = tensor_to_pil(single_tensor, 0).copy()
                
                # Process all tiles for this image
                image_seed = seed + image_idx * 1000
                for tile_idx, pos in enumerate(all_tiles):
                    local_image = self._process_and_blend_tile(
                        tile_idx, pos, single_tensor, local_image,
                        model, positive, negative, vae, image_seed, steps, cfg,
                        sampler_name, scheduler, denoise, tile_width, tile_height,
                        padding, mask_blur, width, height, tiled_decode
                    )
                
                # Send processed image back to master
                try:
                    # Use the estimated remaining to determine if this is the last image
                    is_last = is_last_for_worker
                    run_async_in_server_loop(
                        self._send_full_image_to_master(local_image, image_idx, multi_job_id, 
                                                      master_url, worker_id, is_last),
                        timeout=TILE_SEND_TIMEOUT
                    )
                    # Send heartbeat after processing
                    run_async_in_server_loop(
                        self._send_heartbeat_to_master(multi_job_id, master_url, worker_id),
                        timeout=5.0
                    )
                    if is_last:
                        break
                except Exception as e:
                    log(f"UltimateSDUpscale Worker Dynamic - Error sending image {image_idx}: {e}")
                    # Continue processing other images
            
            # Send final is_last signal
            debug_log(f"UltimateSDUpscale Worker Dynamic - Worker {worker_id} processed {processed_count} images, sending completion signal")
            try:
                run_async_in_server_loop(
                    self._send_worker_complete_signal(multi_job_id, master_url, worker_id),
                    timeout=TILE_SEND_TIMEOUT
                )
            except Exception as e:
                log(f"UltimateSDUpscale Worker Dynamic - Error sending completion signal: {e}")
            
            return (upscaled_image,)
    
    async def _request_image_from_master(self, multi_job_id, master_url, worker_id):
        """Request an image index to process from master in dynamic mode."""
        # Enhanced retries with 404-specific delay, backoff cap, and total timeout to handle init races
        max_retries = 10
        retry_delay = 0.5
        start_time = time.time()
        
        for attempt in range(max_retries):
            # Check total timeout
            if time.time() - start_time > 30:
                log(f"Total request timeout after 30s for worker {worker_id}")
                return None, 0
                
            try:
                session = await get_client_session()
                url = f"{master_url}/distributed/request_image"
                
                async with session.post(url, json={
                    'worker_id': str(worker_id),
                    'multi_job_id': multi_job_id
                }) as response:
                    if response.status == 200:
                        data = await response.json()
                        image_idx = data.get('image_idx')
                        estimated_remaining = data.get('estimated_remaining', 0)
                        return image_idx, estimated_remaining
                    elif response.status == 404:
                        # Special handling for 404 - job not found yet
                        text = await response.text()
                        debug_log(f"Job not found (404), will retry: {text}")
                        await asyncio.sleep(1.0)
                    else:
                        text = await response.text()
                        debug_log(f"Request image failed: {response.status} - {text}")
                        
            except Exception as e:
                if attempt < max_retries - 1:
                    debug_log(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    retry_delay = min(retry_delay, 5.0)  # Cap backoff at 5s
                else:
                    log(f"Failed to request image after {max_retries} attempts: {e}")
                    raise
        
        return None, 0
    
    async def _check_job_status(self, multi_job_id, master_url):
        """Check if job is ready on the master."""
        try:
            session = await get_client_session()
            url = f"{master_url}/distributed/job_status?multi_job_id={multi_job_id}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('ready', False)
                return False
        except Exception as e:
            debug_log(f"Job status check failed: {e}")
            return False
    
    async def _send_full_image_to_master(self, image_pil, image_idx, multi_job_id, 
                                        master_url, worker_id, is_last):
        """Send a processed full image back to master in dynamic mode."""
        # Serialize image to PNG
        byte_io = io.BytesIO()
        image_pil.save(byte_io, format='PNG', compress_level=0)
        byte_io.seek(0)
        
        # Prepare form data
        data = aiohttp.FormData()
        data.add_field('multi_job_id', multi_job_id)
        data.add_field('worker_id', str(worker_id))
        data.add_field('image_idx', str(image_idx))
        data.add_field('is_last', str(is_last))
        data.add_field('full_image', byte_io, filename=f'image_{image_idx}.png', 
                      content_type='image/png')
        
        # Retry logic
        max_retries = 5
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                session = await get_client_session()
                url = f"{master_url}/distributed/tile_complete"
                
                async with session.post(url, data=data) as response:
                    response.raise_for_status()
                    debug_log(f"Successfully sent image {image_idx} to master")
                    return
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    debug_log(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    log(f"Failed to send image {image_idx} after {max_retries} attempts: {e}")
                    raise
    
    async def _send_heartbeat_to_master(self, multi_job_id, master_url, worker_id):
        """Send heartbeat signal to master."""
        try:
            data = {'multi_job_id': multi_job_id, 'worker_id': str(worker_id)}
            session = await get_client_session()
            url = f"{master_url}/distributed/heartbeat"
            
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                debug_log(f"Worker {worker_id} sent heartbeat")
        except Exception as e:
            debug_log(f"Worker {worker_id} heartbeat failed: {e}")
    
    async def _send_worker_complete_signal(self, multi_job_id, master_url, worker_id):
        """Send completion signal to master in dynamic mode."""
        # Send a dummy request with is_last=True
        data = aiohttp.FormData()
        data.add_field('multi_job_id', multi_job_id)
        data.add_field('worker_id', str(worker_id))
        data.add_field('is_last', 'true')
        # Send empty tile data to signal completion
        data.add_field('batch_size', '0')
        
        session = await get_client_session()
        url = f"{master_url}/distributed/tile_complete"
        
        async with session.post(url, data=data) as response:
            response.raise_for_status()
            debug_log(f"Worker {worker_id} sent completion signal")


# Ensure initialization before registering routes
ensure_tile_jobs_initialized()

# API Endpoint for tile completion
@server.PromptServer.instance.routes.post("/distributed/request_image")
async def request_image_endpoint(request):
    """Endpoint for workers to request images in dynamic mode."""
    try:
        data = await request.json()
        worker_id = data.get('worker_id')
        multi_job_id = data.get('multi_job_id')
        
        if not worker_id or not multi_job_id:
            return await handle_api_error(request, "Missing worker_id or multi_job_id", 400)
        
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                if not isinstance(job_data, dict) or 'mode' not in job_data:
                    return await handle_api_error(request, "Invalid job data structure", 500)
                if job_data['mode'] != 'dynamic':
                    return await handle_api_error(request, f"Mode mismatch: expected dynamic, got {job_data['mode']}", 400)
                if 'pending_images' in job_data:
                    pending_queue = job_data['pending_images']
                    try:
                        image_idx = await asyncio.wait_for(pending_queue.get(), timeout=0.1)
                        # Track assigned image
                        if 'assigned_to_workers' in job_data and worker_id in job_data['assigned_to_workers']:
                            job_data['assigned_to_workers'][worker_id].append(image_idx)
                        # Update worker heartbeat
                        if 'worker_status' in job_data:
                            job_data['worker_status'][worker_id] = time.time()
                        # Get estimated remaining count
                        remaining = pending_queue.qsize()  # Approximate
                        debug_log(f"UltimateSDUpscale API - Assigned image {image_idx} to worker {worker_id}")
                        return web.json_response({"image_idx": image_idx, "estimated_remaining": remaining})
                    except asyncio.TimeoutError:
                        return web.json_response({"image_idx": None})
                else:
                    return await handle_api_error(request, "Invalid dynamic mode configuration", 400)
            else:
                return await handle_api_error(request, "Job not found", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.get("/distributed/job_status")
async def job_status_endpoint(request):
    """Endpoint to check if a job is ready."""
    multi_job_id = request.query.get('multi_job_id')
    if not multi_job_id:
        return web.json_response({"ready": False})
    prompt_server = ensure_tile_jobs_initialized()
    async with prompt_server.distributed_tile_jobs_lock:
        job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
        ready = bool(job_data and isinstance(job_data, dict) and 'queue' in job_data)
        return web.json_response({"ready": ready})

@server.PromptServer.instance.routes.post("/distributed/heartbeat")
async def heartbeat_endpoint(request):
    """Endpoint for workers to send heartbeat signals."""
    try:
        data = await request.json()
        worker_id = data.get('worker_id')
        multi_job_id = data.get('multi_job_id')
        
        if not worker_id or not multi_job_id:
            return await handle_api_error(request, "Missing worker_id or multi_job_id", 400)
        
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                if 'worker_status' in job_data:
                    job_data['worker_status'][worker_id] = time.time()
                    debug_log(f"UltimateSDUpscale API - Heartbeat from worker {worker_id}")
                    return web.json_response({"status": "success"})
                else:
                    return await handle_api_error(request, "Worker status tracking not available", 400)
            else:
                return await handle_api_error(request, "Job not found", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)

@server.PromptServer.instance.routes.post("/distributed/tile_complete")
async def tile_complete_endpoint(request):
    """Endpoint for receiving completed tiles from workers."""
    try:
        # Check payload size before processing
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > MAX_PAYLOAD_SIZE:
            return await handle_api_error(request, f"Payload too large: {content_length} bytes (max: {MAX_PAYLOAD_SIZE})", 413)
        
        data = await request.post()
        multi_job_id = data.get('multi_job_id')
        worker_id = data.get('worker_id')
        is_last = data.get('is_last', 'False').lower() == 'true'
        
        if multi_job_id is None or worker_id is None:
            return await handle_api_error(request, "Missing multi_job_id or worker_id", 400)

        prompt_server = ensure_tile_jobs_initialized()
        
        # Check if this is a full image submission (dynamic mode)
        if 'full_image' in data and 'image_idx' in data:
            image_idx = int(data.get('image_idx'))
            is_last = data.get('is_last', 'False').lower() == 'true'
            
            try:
                # Process full image
                img_data = data['full_image'].file.read()
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                debug_log(f"UltimateSDUpscale API - Received full image {image_idx} from worker {worker_id}")
                
                # Put into dynamic mode queue
                async with prompt_server.distributed_tile_jobs_lock:
                    if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                        job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                        if not isinstance(job_data, dict) or 'mode' not in job_data:
                            return await handle_api_error(request, "Invalid job data structure", 500)
                        if job_data['mode'] != 'dynamic':
                            return await handle_api_error(request, f"Mode mismatch: expected dynamic, got {job_data['mode']}", 400)
                        # Check for duplicate image submission
                        if 'completed_images' in job_data and image_idx in job_data['completed_images']:
                            log(f"UltimateSDUpscale API - Duplicate image {image_idx} from {worker_id}, ignoring")
                            return web.json_response({"status": "duplicate"})
                        if 'queue' in job_data:
                            await job_data['queue'].put({
                                'worker_id': worker_id,
                                'image_idx': image_idx,
                                'image': img,
                                'is_last': is_last
                            })
                            return web.json_response({"status": "success"})
                        else:
                            return await handle_api_error(request, "Invalid dynamic mode configuration", 400)
                    else:
                        return await handle_api_error(request, "Job not found", 404)
            except Exception as e:
                log(f"Error processing full image from worker {worker_id}: {e}")
                return await handle_api_error(request, f"Image processing error: {e}", 400)
        
        # Check for batch mode
        batch_size = int(data.get('batch_size', 0))
        tiles = []
        
        # Handle completion signals (batch_size=0 with is_last=True)
        if batch_size == 0 and is_last:
            async with prompt_server.distributed_tile_jobs_lock:
                if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                    job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                    if isinstance(job_data, dict) and 'queue' in job_data:
                        await job_data['queue'].put({
                            'worker_id': worker_id,
                            'is_last': True,
                            'tiles': []
                        })
                        debug_log(f"Received completion signal from worker {worker_id}")
                        return web.json_response({"status": "success"})
                    else:
                        return await handle_api_error(request, "Invalid job structure", 500)
                else:
                    return await handle_api_error(request, "Job not found", 404)
        
        if batch_size > 0:
            # Batch mode: Extract multiple tiles
            padding = int(data.get('padding', 32))
            debug_log(f"UltimateSDUpscale - tile_complete batch - job_id: {multi_job_id}, worker: {worker_id}, batch_size: {batch_size}")
            
            # Check for JSON metadata
            metadata_field = data.get('tiles_metadata')
            if metadata_field:
                # New JSON metadata format
                try:
                    # Handle different types of metadata field
                    if hasattr(metadata_field, 'file'):
                        # File-like object
                        metadata_str = metadata_field.file.read().decode('utf-8')
                    elif isinstance(metadata_field, (bytes, bytearray)):
                        # Direct bytes/bytearray
                        metadata_str = metadata_field.decode('utf-8')
                    else:
                        # String
                        metadata_str = str(metadata_field)
                    
                    metadata = json.loads(metadata_str)
                    if len(metadata) != batch_size:
                        return await handle_api_error(request, "Metadata length mismatch", 400)
                    
                    tile_data_list = []  # Temporary list to collect and sort tiles
                    
                    for i in range(batch_size):
                        tile_field = data.get(f'tile_{i}')
                        if tile_field is None:
                            log(f"Missing tile_{i} from worker {worker_id}, skipping")
                            continue
                        
                        try:
                            # Process image
                            img_data = tile_field.file.read()
                            img = Image.open(io.BytesIO(img_data)).convert("RGB")
                            img_np = np.array(img).astype(np.float32) / 255.0
                            tensor = torch.from_numpy(img_np)[None,]
                            
                            # Get metadata for this tile
                            if i < len(metadata):
                                tile_data = metadata[i]
                                tile_idx = tile_data.get('tile_idx', i)
                                
                                # Validate order
                                if i != tile_idx % batch_size:
                                    debug_log(f"Warning: Tile order mismatch at position {i}, expected tile_idx {tile_idx}")
                                
                                tile_info = {
                                    'tensor': tensor,
                                    'tile_idx': tile_idx,
                                    'x': tile_data['x'],
                                    'y': tile_data['y'],
                                    'extracted_width': tile_data['extracted_width'],
                                    'extracted_height': tile_data['extracted_height'],
                                    'padding': padding,
                                    'batch_idx': tile_data.get('batch_idx', 0),
                                    'global_idx': tile_data.get('global_idx', tile_idx)
                                }
                                tile_data_list.append(tile_info)
                            else:
                                log(f"Missing metadata for tile {i} from worker {worker_id}, skipping")
                                continue
                                
                        except Exception as e:
                            log(f"Error processing tile {i} from worker {worker_id}: {e}, skipping this tile")
                            continue
                    
                    # Sort tiles by tile_idx to ensure correct order
                    tile_data_list.sort(key=lambda x: x['tile_idx'])
                    tiles.extend(tile_data_list)
                    
                    if tile_data_list:
                        debug_log(f"Reordered {len(tile_data_list)} tiles based on tile_idx")
                except Exception as e:
                    log(f"Error processing JSON metadata from worker {worker_id}: {e}")
                    return await handle_api_error(request, f"Metadata processing error: {e}", 400)
            else:
                # Legacy format: individual fields per tile (backward compatibility)
                debug_log(f"WARNING: Worker {worker_id} using legacy field format. Please update to use JSON metadata.")
                
                for i in range(batch_size):
                    tile_field = data.get(f'tile_{i}')
                    if tile_field is None:
                        log(f"Missing tile_{i} from worker {worker_id}, skipping")
                        continue
                    
                    # Get metadata for this tile
                    tile_idx = int(data.get(f'tile_{i}_idx', i))
                    x = int(data.get(f'tile_{i}_x', 0))
                    y = int(data.get(f'tile_{i}_y', 0))
                    extracted_width = int(data.get(f'tile_{i}_width', 512))
                    extracted_height = int(data.get(f'tile_{i}_height', 512))
                    
                    try:
                        # Process image
                        img_data = tile_field.file.read()
                        img = Image.open(io.BytesIO(img_data)).convert("RGB")
                        img_np = np.array(img).astype(np.float32) / 255.0
                        tensor = torch.from_numpy(img_np)[None,]
                        
                        tiles.append({
                            'tensor': tensor,
                            'tile_idx': tile_idx,
                            'x': x,
                            'y': y,
                            'extracted_width': extracted_width,
                            'extracted_height': extracted_height,
                            'padding': padding
                        })
                    except Exception as e:
                        log(f"Error processing tile {i} from worker {worker_id}: {e}, skipping this tile")
                        continue
        else:
            # Single tile mode (backward compatibility)
            image_file = data.get('image')
            if not image_file:
                return await handle_api_error(request, "Missing image data", 400)
                
            tile_idx = int(data.get('tile_idx', 0))
            x = int(data.get('x', 0))
            y = int(data.get('y', 0))
            extracted_width = int(data.get('extracted_width', 512))
            extracted_height = int(data.get('extracted_height', 512))
            padding = int(data.get('padding', 32))
            
            debug_log(f"UltimateSDUpscale - tile_complete single - job_id: {multi_job_id}, worker: {worker_id}, tile: {tile_idx}")
            
            try:
                # Process image
                img_data = image_file.file.read()
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                tensor = torch.from_numpy(img_np)[None,]
                
                tiles = [{
                    'tensor': tensor,
                    'tile_idx': tile_idx,
                    'x': x,
                    'y': y,
                    'extracted_width': extracted_width,
                    'extracted_height': extracted_height,
                    'padding': padding,
                    'batch_idx': 0,  # Default for legacy single-tile
                    'global_idx': tile_idx  # Default for legacy single-tile
                }]
            except Exception as e:
                log(f"Error processing tile from worker {worker_id}: {e}")
                return await handle_api_error(request, f"Tile processing error: {e}", 400)

        # Put tiles into queue
        async with prompt_server.distributed_tile_jobs_lock:
            debug_log(f"UltimateSDUpscale - tile_complete: Checking distributed_pending_tile_jobs for job {multi_job_id}")
            debug_log(f"UltimateSDUpscale - tile_complete: Current jobs in distributed_pending_tile_jobs: {list(prompt_server.distributed_pending_tile_jobs.keys())}")
            
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                job_data = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                if not isinstance(job_data, dict) or 'mode' not in job_data:
                    return await handle_api_error(request, "Invalid job data structure", 500)
                if job_data['mode'] != 'static':
                    return await handle_api_error(request, f"Mode mismatch: expected static, got {job_data['mode']}", 400)
                
                q = job_data['queue']
                if batch_size > 0:
                    # Put batch as single item  
                    await q.put({
                        'worker_id': worker_id,
                        'tiles': tiles,
                        'is_last': is_last
                    })
                    debug_log(f"UltimateSDUpscale - Received batch of {len(tiles)} tiles for job {multi_job_id} from worker {worker_id}")
                else:
                    # Put single tile (backward compat)
                    tile_data = tiles[0]
                    await q.put({
                        'tensor': tile_data['tensor'],
                        'worker_id': worker_id,
                        'tile_idx': tile_data['tile_idx'],
                        'x': tile_data['x'],
                        'y': tile_data['y'],
                        'extracted_width': tile_data['extracted_width'],
                        'extracted_height': tile_data['extracted_height'],
                        'padding': tile_data['padding'],
                        'is_last': is_last
                    })
                    debug_log(f"UltimateSDUpscale - Received single tile {tile_data['tile_idx']} for job {multi_job_id} from worker {worker_id}")
                    
                return web.json_response({"status": "success"})
            else:
                debug_log(f"UltimateSDUpscale - API Error: Job {multi_job_id} not found in distributed_pending_tile_jobs")
                debug_log(f"UltimateSDUpscale - API Error: Available jobs: {list(prompt_server.distributed_pending_tile_jobs.keys())}")
                return await handle_api_error(request, "Job not found or already complete", 404)
    except Exception as e:
        return await handle_api_error(request, e, 500)


# Node registration
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscaleDistributed": UltimateSDUpscaleDistributed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscaleDistributed": "Ultimate SD Upscale Distributed (No Upscale)",
}