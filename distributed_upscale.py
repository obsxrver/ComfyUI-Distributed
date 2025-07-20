# Import statements (original + new)
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

# Import for controller support
from .utils.usdu_utils import crop_cond, clone_control_chain, clone_conditioning, ensure_tile_jobs_initialized


# Make MAX_BATCH configurable
MAX_BATCH = int(os.environ.get('COMFYUI_MAX_BATCH', '20'))

# Heartbeat timeout in seconds (configurable via env var, default 120s to cover typical image processing)
HEARTBEAT_TIMEOUT = int(os.environ.get('COMFYUI_HEARTBEAT_TIMEOUT', '120'))

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
        
        mode = self._determine_processing_mode(batch_size, num_workers, dynamic_threshold)
            
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
        assigned_global_indices = self._get_worker_global_indices(total_tiles, enabled_workers, worker_id)
        if not assigned_global_indices:
            return (upscaled_image,)
        
        debug_log(f"UltimateSDUpscale Worker - Worker {worker_id} processing {len(assigned_global_indices)} tiles from batch of {batch_size}")
        
        # Process tiles SYNCHRONOUSLY
        processed_tiles = []
        # Track which batch indices we've already pre-sliced conditioning for
        sliced_conditioning_cache = {}
        
        for global_idx in assigned_global_indices:
            # Calculate which image and tile this corresponds to
            batch_idx = global_idx // num_tiles_per_image
            tile_idx = global_idx % num_tiles_per_image
            
            # Skip if batch_idx is out of range
            if batch_idx >= upscaled_image.shape[0]:
                debug_log(f"Warning: Calculated batch_idx {batch_idx} exceeds batch size {upscaled_image.shape[0]}")
                continue
            
            # Pre-slice conditioning once per image (cache it)
            if batch_idx not in sliced_conditioning_cache:
                positive_sliced = clone_conditioning(positive)
                negative_sliced = clone_conditioning(negative)
                for cond_list in [positive_sliced, negative_sliced]:
                    for i in range(len(cond_list)):
                        emb, cond_dict = cond_list[i]
                        if emb.shape[0] > 1:
                            cond_list[i][0] = emb[batch_idx:batch_idx+1]
                        if 'control' in cond_dict:
                            control = cond_dict['control']
                            while control is not None:
                                hint = control.cond_hint_original
                                if hint.shape[0] > 1:
                                    control.cond_hint_original = hint[batch_idx:batch_idx+1]
                                control = control.previous_controlnet
                        if 'mask' in cond_dict and cond_dict['mask'].shape[0] > 1:
                            cond_dict['mask'] = cond_dict['mask'][batch_idx:batch_idx+1]
                sliced_conditioning_cache[batch_idx] = (positive_sliced, negative_sliced)
            else:
                positive_sliced, negative_sliced = sliced_conditioning_cache[batch_idx]
                
            x, y = all_tiles[tile_idx]
            
            # Extract tile from the specific image in the batch
            tile_tensor, x1, y1, ew, eh = self.extract_tile_with_padding(
                upscaled_image[batch_idx:batch_idx+1], x, y, tile_width, tile_height, padding
            )
            
            # Process tile through SD with unique seed
            image_seed = seed + batch_idx * 1000
            processed_tile = self.process_tile(tile_tensor, model, positive_sliced, negative_sliced, vae,
                                             image_seed + tile_idx, steps, cfg, sampler_name, 
                                             scheduler, denoise, tiled_decode, batch_idx=batch_idx,
                                             region=(x1, y1, x1 + ew, y1 + eh), image_size=(width, height))
            
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
        mode = self._determine_processing_mode(batch_size, num_workers, dynamic_threshold)
        
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
        # Track which batch indices we've already pre-sliced conditioning for
        sliced_conditioning_cache = {}
        
        for global_idx in master_global_indices:
            # Calculate which image and tile this corresponds to
            batch_idx = global_idx // num_tiles_per_image
            tile_idx = global_idx % num_tiles_per_image
            
            # Skip if batch_idx is out of range
            if batch_idx >= len(result_images):
                debug_log(f"Warning: Calculated batch_idx {batch_idx} exceeds result_images length {len(result_images)}")
                continue
            
            # Pre-slice conditioning once per image (cache it)
            if batch_idx not in sliced_conditioning_cache:
                positive_sliced = clone_conditioning(positive)
                negative_sliced = clone_conditioning(negative)
                for cond_list in [positive_sliced, negative_sliced]:
                    for i in range(len(cond_list)):
                        emb, cond_dict = cond_list[i]
                        if emb.shape[0] > 1:
                            cond_list[i][0] = emb[batch_idx:batch_idx+1]
                        if 'control' in cond_dict:
                            control = cond_dict['control']
                            while control is not None:
                                hint = control.cond_hint_original
                                if hint.shape[0] > 1:
                                    control.cond_hint_original = hint[batch_idx:batch_idx+1]
                                control = control.previous_controlnet
                        if 'mask' in cond_dict and cond_dict['mask'].shape[0] > 1:
                            cond_dict['mask'] = cond_dict['mask'][batch_idx:batch_idx+1]
                sliced_conditioning_cache[batch_idx] = (positive_sliced, negative_sliced)
            else:
                positive_sliced, negative_sliced = sliced_conditioning_cache[batch_idx]
                
            # Use unique seed per image
            image_seed = seed + batch_idx * 1000
            
            result_images[batch_idx] = self._process_and_blend_tile(
                tile_idx, all_tiles[tile_idx], upscaled_image[batch_idx:batch_idx+1], result_images[batch_idx],
                model, positive_sliced, negative_sliced, vae, image_seed, steps, cfg,
                sampler_name, scheduler, denoise, tile_width, tile_height,
                padding, mask_blur, width, height, tiled_decode, batch_idx=batch_idx
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
    
    def _get_worker_global_indices(self, total_tiles, enabled_workers, worker_id):
        """Calculate which global tile indices are assigned to a specific worker in flattened mode."""
        num_workers = len(enabled_workers)
        
        debug_log(f"UltimateSDUpscale Worker - Worker ID: {worker_id}, Enabled workers: {enabled_workers}")
        
        try:
            worker_index = enabled_workers.index(worker_id)
        except ValueError:
            log(f"UltimateSDUpscale Worker - Worker {worker_id} not found in enabled list {enabled_workers}")
            return []
        
        global_indices = list(range(total_tiles))
        all_assignments = self._distribute_items(global_indices, num_workers + 1)
        return all_assignments[worker_index + 1]  # +1 because 0 is master
    
    def _get_master_global_indices(self, total_tiles, num_workers):
        """Calculate which global tile indices are assigned to the master in flattened mode."""
        global_indices = list(range(total_tiles))
        all_assignments = self._distribute_items(global_indices, num_workers + 1)
        return all_assignments[0]
    
    def _process_and_blend_tile(self, tile_idx, tile_pos, upscaled_image, result_image,
                               model, positive, negative, vae, seed, steps, cfg,
                               sampler_name, scheduler, denoise, tile_width, tile_height,
                               padding, mask_blur, image_width, image_height, tiled_decode, batch_idx: int = 0):
        """Process a single tile and blend it into the result image."""
        x, y = tile_pos
        
        # Extract and process tile
        tile_tensor, x1, y1, ew, eh = self.extract_tile_with_padding(
            upscaled_image, x, y, tile_width, tile_height, padding
        )
        
        processed_tile = self.process_tile(tile_tensor, model, positive, negative, vae,
                                         seed + tile_idx, steps, cfg, sampler_name, 
                                         scheduler, denoise, tiled_decode, batch_idx=batch_idx,
                                         region=(x1, y1, x1 + ew, y1 + eh), image_size=(image_width, image_height))
        
        # Convert and blend
        processed_pil = tensor_to_pil(processed_tile, 0)
        # Create mask for this specific tile
        tile_mask = self.create_tile_mask(image_width, image_height, x, y, tile_width, tile_height, mask_blur)
        # Use extraction position and size for blending
        result_image = self.blend_tile(result_image, processed_pil, 
                                     x1, y1, (ew, eh), tile_mask, padding)
        
        return result_image
    
    async def _init_job_queue_dynamic(self, multi_job_id, batch_size, assigned_to_workers=None, worker_status=None, all_indices=None):
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
                # Initialize pending images queue with all indices
                pending_images = prompt_server.distributed_pending_tile_jobs[multi_job_id]['pending_images']
                for i in all_indices:
                    await pending_images.put(i)
                debug_log(f"UltimateSDUpscale Master Dynamic - Initialized job queue with {len(all_indices)} pending images for all participants")
            else:
                debug_log(f"UltimateSDUpscale Master Dynamic - Queue already exists for {multi_job_id}")
    
    async def _init_job_queue(self, multi_job_id):
        """Initialize the job queue for collecting tiles."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id not in prompt_server.distributed_pending_tile_jobs:
                prompt_server.distributed_pending_tile_jobs[multi_job_id] = {
                    'queue': asyncio.Queue(),
                    'mode': 'static',
                    'completed_tiles': {}  # For static mode data
                }
                debug_log(f"UltimateSDUpscale Master - Initialized job queue for {multi_job_id}")
    
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
                
            except asyncio.TimeoutError:
                log(f"UltimateSDUpscale Master - Timeout waiting for tiles")
                break
        
        debug_log(f"UltimateSDUpscale Master - Collection complete. Got {len(collected_tiles)} tiles from {len(workers_done)} workers")
        
        # Clean up job queue
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                del prompt_server.distributed_pending_tile_jobs[multi_job_id]
        
        return collected_tiles
    
    async def _mark_image_completed(self, multi_job_id, image_idx, image_pil):
        """Mark an image as completed in the job data."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if job_data and 'completed_images' in job_data:
                job_data['completed_images'][image_idx] = image_pil

    async def _async_collect_dynamic_images(self, multi_job_id, remaining_to_collect, num_workers, batch_size, master_processed_count):
        """Collect remaining processed images from workers."""
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
        collected_count = 0
        timeout = TILE_WAIT_TIMEOUT * 2  # Longer timeout for full images
        last_heartbeat_check = time.time()
        
        debug_log(f"UltimateSDUpscale Master Dynamic - Waiting for {remaining_to_collect} more images from {num_workers} workers")
        
        while collected_count < remaining_to_collect and len(workers_done) < num_workers:
            try:
                result = await asyncio.wait_for(q.get(), timeout=10.0)  # Shorter timeout for regular checks
                worker_id = result['worker_id']
                is_last = result.get('is_last', False)
                
                if 'image_idx' in result and 'image' in result:
                    image_idx = result['image_idx']
                    image_pil = result['image']
                    completed_images[image_idx] = image_pil
                    collected_count += 1
                    debug_log(f"UltimateSDUpscale Master Dynamic - Received image {image_idx} from worker {worker_id}")
                
                if is_last:
                    workers_done.add(worker_id)
                    debug_log(f"UltimateSDUpscale Master Dynamic - Worker {worker_id} completed")
                    
            except asyncio.TimeoutError:
                # Check for worker timeouts
                current_time = time.time()
                if current_time - last_heartbeat_check >= 10.0:
                    async with prompt_server.distributed_tile_jobs_lock:
                        if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                            job_data_check = prompt_server.distributed_pending_tile_jobs[multi_job_id]
                            # Check for timed out workers (60 second timeout)
                            for worker, last_heartbeat in list(job_data_check.get('worker_status', {}).items()):
                                if current_time - last_heartbeat > HEARTBEAT_TIMEOUT:
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
                            for idx in range(batch_size):
                                if idx not in completed_images:
                                    await pending_queue.put(idx)
                                    debug_log(f"UltimateSDUpscale Master Dynamic - Requeued image {idx}")
                    break
        
        debug_log(f"UltimateSDUpscale Master Dynamic - Collection complete. Got {collected_count} images from workers")
        
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
                     scheduler: str, denoise: float, tiled_decode: bool = False,
                     batch_idx: int = 0, region: Tuple[int, int, int, int] = None,
                     image_size: Tuple[int, int] = None) -> torch.Tensor:
        """Process a single tile through SD sampling. 
        Note: positive and negative should already be pre-sliced for the current batch_idx."""
        debug_log(f"[process_tile] Processing tile for batch_idx={batch_idx}, seed={seed}, region={region}")
        
        
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
        
        # Clone conditioning per tile (shares models, clones hints for cropping)
        positive_tile = clone_conditioning(positive, clone_hints=True)
        negative_tile = clone_conditioning(negative, clone_hints=True)
        
        # Crop conditioning to tile region if provided (assumes hints at image resolution)
        if region is not None and image_size is not None:
            init_size = image_size  # (width, height) of full image
            canvas_size = image_size
            tile_size = (tile_tensor.shape[2], tile_tensor.shape[1])  # (width, height)
            w_pad = 0  # No extra pad needed; region already includes padding
            h_pad = 0
            positive_cropped = crop_cond(positive_tile, region, init_size, canvas_size, tile_size, w_pad, h_pad)
            negative_cropped = crop_cond(negative_tile, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        else:
            # No region cropping needed, use cloned conditioning as-is
            positive_cropped = positive_tile
            negative_cropped = negative_tile
        
        # Encode to latent (always non-tiled, matching original node)
        latent = VAEEncode().encode(vae, clean_tensor)[0]
        
        # Sample
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                positive_cropped, negative_cropped, latent, denoise=denoise)[0]
        
        # Decode back to image
        if tiled_decode and tiled_vae_available:
            image = VAEDecodeTiled().decode(vae, samples, tile_size=512)[0]
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
                  mask: Image.Image, padding: int) -> Image.Image:
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
            
            
            # Retry logic with exponential backoff
            max_retries = 5
            retry_delay = 0.5
            
            for attempt in range(max_retries):
                try:
                    session = await get_client_session()
                    url = f"{master_url}/distributed/tile_complete"
                    
                    async with session.post(url, data=data) as response:
                        response.raise_for_status()
                        break  # Success, move to next chunk
                        
                except Exception as e:
                    if attempt < max_retries - 1:
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
            debug_log(f"[process_single_gpu] Processing batch_idx {batch_idx}")
            # Pre-slice conditioning once per image (not per tile)
            positive_sliced = clone_conditioning(positive)
            negative_sliced = clone_conditioning(negative)
            for cond_list in [positive_sliced, negative_sliced]:
                for i in range(len(cond_list)):
                    emb, cond_dict = cond_list[i]
                    if emb.shape[0] > 1:
                        cond_list[i][0] = emb[batch_idx:batch_idx+1]
                    if 'control' in cond_dict:
                        control = cond_dict['control']
                        while control is not None:
                            hint = control.cond_hint_original
                            if hint.shape[0] > 1:
                                control.cond_hint_original = hint[batch_idx:batch_idx+1]
                            control = control.previous_controlnet
                    if 'mask' in cond_dict and cond_dict['mask'].shape[0] > 1:
                        cond_dict['mask'] = cond_dict['mask'][batch_idx:batch_idx+1]
            
            # Process each tile for this image with pre-sliced conditioning
            for tile_idx, tile_pos in enumerate(all_tiles):
                # Use unique seed per image
                image_seed = seed + batch_idx * 1000
                result_images[batch_idx] = self._process_and_blend_tile(
                    tile_idx, tile_pos, upscaled_image[batch_idx:batch_idx+1], result_images[batch_idx],
                    model, positive_sliced, negative_sliced, vae, image_seed, steps, cfg,
                    sampler_name, scheduler, denoise, tile_width, tile_height,
                    padding, mask_blur, width, height, tiled_decode, batch_idx=batch_idx
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
        """Dynamic mode for large batches - assigns whole images to workers dynamically, including master."""
        # Get batch size and dimensions
        batch_size, height, width, _ = upscaled_image.shape
        num_workers = len(enabled_workers)
        
        debug_log(f"UltimateSDUpscale Master Dynamic - Processing batch of {batch_size} images with {num_workers} workers")
        
        # No fixed share - all images are dynamic
        all_indices = list(range(batch_size))
        
        log(f"Processing {batch_size} images dynamically across master + {num_workers} workers.")
        
        # Calculate tiles for processing
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height, force_uniform_tiles)
        
        # Initialize job queue for communication
        try:
            assigned_to_workers = {w: [] for w in enabled_workers}
            worker_status = {w: time.time() for w in enabled_workers}
            
            run_async_in_server_loop(
                self._init_job_queue_dynamic(multi_job_id, batch_size, assigned_to_workers, worker_status, all_indices),
                timeout=2.0
            )
        except Exception as e:
            debug_log(f"UltimateSDUpscale Master Dynamic - Queue initialization error: {e}")
            raise RuntimeError(f"Failed to initialize dynamic mode queue: {e}")
        
        # Convert batch to PIL list
        result_images = [tensor_to_pil(upscaled_image[b:b+1], 0).convert('RGB').copy() for b in range(batch_size)]
        
        # Process images dynamically with master participating
        prompt_server = ensure_tile_jobs_initialized()
        processed_count = 0
        consecutive_retries = 0
        max_consecutive_retries = 10
        
        # Process loop - master pulls from queue and processes synchronously
        while processed_count < batch_size:
            # Try to get an image to process
            image_idx = run_async_in_server_loop(
                self._get_next_image_index(multi_job_id),
                timeout=5.0  # Short timeout to allow frequent checks
            )

            if image_idx is not None:
                # Reset retry counter and process locally
                consecutive_retries = 0
                debug_log(f"Master processing image {image_idx} dynamically")
                processed_count += 1

                # Process locally
                single_tensor = upscaled_image[image_idx:image_idx+1]
                local_image = result_images[image_idx]
                image_seed = seed + image_idx * 1000
                
                # Pre-slice conditioning once per image (not per tile)
                positive_sliced = clone_conditioning(positive)
                negative_sliced = clone_conditioning(negative)
                for cond_list in [positive_sliced, negative_sliced]:
                    for i in range(len(cond_list)):
                        emb, cond_dict = cond_list[i]
                        if emb.shape[0] > 1:
                            cond_list[i][0] = emb[image_idx:image_idx+1]
                        if 'control' in cond_dict:
                            control = cond_dict['control']
                            while control is not None:
                                hint = control.cond_hint_original
                                if hint.shape[0] > 1:
                                    control.cond_hint_original = hint[image_idx:image_idx+1]
                                control = control.previous_controlnet
                        if 'mask' in cond_dict and cond_dict['mask'].shape[0] > 1:
                            cond_dict['mask'] = cond_dict['mask'][image_idx:image_idx+1]
                
                for tile_idx, pos in enumerate(all_tiles):
                    local_image = self._process_and_blend_tile(
                        tile_idx, pos, single_tensor, local_image,
                        model, positive_sliced, negative_sliced, vae, image_seed, steps, cfg,
                        sampler_name, scheduler, denoise, tile_width, tile_height,
                        padding, mask_blur, width, height, tiled_decode, batch_idx=image_idx
                    )
                    
                    # Yield after each tile to minimize worker downtime
                    run_async_in_server_loop(self._async_yield(), timeout=0.1)
                    # Note: No per-tile drain here – that's what makes this "per-image"
                
                result_images[image_idx] = local_image
                
                # Mark as completed
                run_async_in_server_loop(
                    self._mark_image_completed(multi_job_id, image_idx, local_image),
                    timeout=5.0
                )
                
                # NEW: Drain after the full image is marked complete (catches workers who finished during master's processing)
                drained_count = run_async_in_server_loop(
                    self._drain_worker_results_queue(multi_job_id),
                    timeout=5.0
                )
                
                if drained_count > 0:
                    debug_log(f"Drained {drained_count} worker images after master's image {image_idx}")
                
                # NEW: Log overall progress (includes master's image + any drained workers)
                completed_now = run_async_in_server_loop(
                    self._get_total_completed_count(multi_job_id),
                    timeout=1.0
                )
                log(f"Progress: {completed_now}/{batch_size} images completed")
                
                # Yield to allow workers to get new images after completing one
                run_async_in_server_loop(self._async_yield(), timeout=0.1)
            else:
                # Queue empty: collect any queued worker results to update progress
                drained_count = run_async_in_server_loop(
                    self._drain_worker_results_queue(multi_job_id),
                    timeout=5.0
                )
                run_async_in_server_loop(self._async_yield(), timeout=0.1)  # Yield after drain
                
                # Check for timed out workers and requeue their images
                requeued_count = run_async_in_server_loop(
                    self._check_and_requeue_timed_out_workers(multi_job_id, batch_size),
                    timeout=5.0
                )
                run_async_in_server_loop(self._async_yield(), timeout=0.1)  # Yield after requeue
                
                if requeued_count > 0:
                    log(f"Requeued {requeued_count} images from timed out workers")
                    consecutive_retries = 0  # Reset since we have work to do
                    continue

                # Now check total completed (includes newly collected)
                completed_now = run_async_in_server_loop(
                    self._get_total_completed_count(multi_job_id),
                    timeout=1.0
                )
                
                log(f"Progress: {completed_now}/{batch_size} images completed")
                
                if completed_now >= batch_size:
                    break

                run_async_in_server_loop(self._async_yield(), timeout=0.1)  # Yield before pending check
                
                # Check if there are pending images in the queue (could be requeued)
                pending_count = run_async_in_server_loop(
                    self._get_pending_count(multi_job_id),
                    timeout=1.0
                )
                
                if pending_count > 0:
                    consecutive_retries = 0  # Reset retries since there's work to do
                    continue

                consecutive_retries += 1
                if consecutive_retries >= max_consecutive_retries:
                    log(f"Max retries ({max_consecutive_retries}) reached. Forcing collection of remaining results.")
                    break  # Force exit to collection phase

                debug_log("Waiting for workers")
                # Use async sleep to allow event loop to process worker requests
                run_async_in_server_loop(asyncio.sleep(2), timeout=3.0)
        
        debug_log(f"Master processed {processed_count} images locally")
        
        # Get all completed images to check what needs to be collected
        all_completed = run_async_in_server_loop(
            self._get_all_completed_images(multi_job_id),
            timeout=5.0
        )
        
        # Calculate how many we still need to collect
        remaining_to_collect = batch_size - len(all_completed)
        
        if remaining_to_collect > 0:
            debug_log(f"Waiting for {remaining_to_collect} more images from workers")
            collected_images = run_async_in_server_loop(
                self._async_collect_dynamic_images(multi_job_id, remaining_to_collect, num_workers, batch_size, processed_count),
                timeout=TILE_COLLECTION_TIMEOUT * 2
            )
            
            # Merge collected with already completed
            all_completed.update(collected_images)
        
        # Update result images with all completed images
        for idx, processed_img in all_completed.items():
            if idx < batch_size:
                result_images[idx] = processed_img
        
        # Convert back to tensor
        result_tensor = torch.cat([pil_to_tensor(img) for img in result_images], dim=0) if batch_size > 1 else pil_to_tensor(result_images[0])
        if upscaled_image.is_cuda:
            result_tensor = result_tensor.cuda()
        
        debug_log(f"UltimateSDUpscale Master Dynamic - Job {multi_job_id} complete")
        log(f"Completed processing all {batch_size} images")
        return (result_tensor,)
    
    
    async def _async_yield(self):
        """Simple async yield to allow event loop processing."""
        await asyncio.sleep(0)
    
    async def _get_next_image_index(self, multi_job_id):
        """Get next image index from pending queue for master."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if not job_data or 'pending_images' not in job_data:
                return None
            try:
                image_idx = await asyncio.wait_for(job_data['pending_images'].get(), timeout=1.0)
                return image_idx
            except asyncio.TimeoutError:
                return None
    
    async def _get_completed_count(self, multi_job_id):
        """Get count of completed images from workers."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if job_data and 'completed_images' in job_data:
                # Count only worker-completed images (not master's)
                completed = job_data['completed_images']
                worker_count = len([idx for idx in completed if isinstance(completed[idx], Image.Image)])
                return worker_count
            return 0
    
    async def _get_total_completed_count(self, multi_job_id):
        """Get total count of all completed images (master + workers)."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if job_data and 'completed_images' in job_data:
                return len(job_data['completed_images'])
            return 0
    
    async def _get_all_completed_images(self, multi_job_id):
        """Get all completed images."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if job_data and 'completed_images' in job_data:
                return job_data['completed_images'].copy()
            return {}
    
    async def _get_pending_count(self, multi_job_id):
        """Get count of pending images in the queue."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if job_data and 'pending_images' in job_data:
                return job_data['pending_images'].qsize()
            return 0
    
    async def _drain_worker_results_queue(self, multi_job_id):
        """Drain pending worker results from queue and update completed_images. Returns count of drained images."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if not job_data or 'queue' not in job_data or 'completed_images' not in job_data:
                return 0
            q = job_data['queue']
            completed_images = job_data['completed_images']

            collected = 0
            while not q.empty():
                try:
                    result = await asyncio.wait_for(q.get(), timeout=0.1)
                    worker_id = result['worker_id']
                    is_last = result.get('is_last', False)

                    if 'image_idx' in result and 'image' in result:
                        image_idx = result['image_idx']
                        image_pil = result['image']
                        if image_idx not in completed_images:
                            completed_images[image_idx] = image_pil
                            collected += 1
                            debug_log(f"Drained image {image_idx} from worker {worker_id}")

                    if is_last:
                        # Optional: track worker completion if needed
                        pass
                except asyncio.TimeoutError:
                    break  # No more immediately available

            if collected > 0:
                debug_log(f"Drained {collected} worker images during retry")
            
            return collected
    
    async def _check_and_requeue_timed_out_workers(self, multi_job_id, batch_size):
        """Check for timed out workers and requeue their assigned images. Returns count of requeued images."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            job_data = prompt_server.distributed_pending_tile_jobs.get(multi_job_id)
            if not job_data:
                return 0
                
            current_time = time.time()
            requeued_count = 0
            completed_images = job_data.get('completed_images', {})
            
            # Check for timed out workers (60 second timeout)
            for worker, last_heartbeat in list(job_data.get('worker_status', {}).items()):
                if current_time - last_heartbeat > HEARTBEAT_TIMEOUT:
                    log(f"Worker {worker} timed out")
                    # Requeue assigned images from this worker
                    for idx in job_data.get('assigned_to_workers', {}).get(worker, []):
                        if idx not in completed_images:
                            await job_data['pending_images'].put(idx)
                            requeued_count += 1
                            debug_log(f"Requeued image {idx} from timed out worker {worker}")
                    # Clean up worker tracking
                    if 'worker_status' in job_data:
                        del job_data['worker_status'][worker]
                    if 'assigned_to_workers' in job_data:
                        job_data['assigned_to_workers'][worker] = []
            
            return requeued_count
    
    
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
                
                # Pre-slice conditioning once per image (not per tile)
                positive_sliced = clone_conditioning(positive)
                negative_sliced = clone_conditioning(negative)
                for cond_list in [positive_sliced, negative_sliced]:
                    for i in range(len(cond_list)):
                        emb, cond_dict = cond_list[i]
                        if emb.shape[0] > 1:
                            cond_list[i][0] = emb[image_idx:image_idx+1]
                        if 'control' in cond_dict:
                            control = cond_dict['control']
                            while control is not None:
                                hint = control.cond_hint_original
                                if hint.shape[0] > 1:
                                    control.cond_hint_original = hint[image_idx:image_idx+1]
                                control = control.previous_controlnet
                        if 'mask' in cond_dict and cond_dict['mask'].shape[0] > 1:
                            cond_dict['mask'] = cond_dict['mask'][image_idx:image_idx+1]
                
                for tile_idx, pos in enumerate(all_tiles):
                    local_image = self._process_and_blend_tile(
                        tile_idx, pos, single_tensor, local_image,
                        model, positive_sliced, negative_sliced, vae, image_seed, steps, cfg,
                        sampler_name, scheduler, denoise, tile_width, tile_height,
                        padding, mask_blur, width, height, tiled_decode, batch_idx=image_idx
                    )
                    run_async_in_server_loop(
                        self._send_heartbeat_to_master(multi_job_id, master_url, worker_id),
                        timeout=5.0
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
        except Exception as e:
            pass
    
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

    def _determine_processing_mode(self, batch_size: int, num_workers: int, dynamic_threshold: int) -> str:
        """Determines the processing mode based on batch size and worker count."""
        if num_workers == 0:
            return "single_gpu"
        if batch_size <= dynamic_threshold:
            return "static"
        return "dynamic"

    def _distribute_items(self, items: list, num_participants: int) -> List[List[any]]:
        """Distributes a list of items among N participants."""
        if num_participants == 0:
            return [items] # All items for the single participant (master)

        items_per_participant = len(items) // num_participants
        remainder = len(items) % num_participants
        
        assignments = []
        start_idx = 0
        for i in range(num_participants):
            count = items_per_participant + (1 if i < remainder else 0)
            end_idx = start_idx + count
            assignments.append(items[start_idx:end_idx])
            start_idx = end_idx
        
        return assignments

# Ensure initialization before registering routes
ensure_tile_jobs_initialized()

# Node registration
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscaleDistributed": UltimateSDUpscaleDistributed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscaleDistributed": "Ultimate SD Upscale Distributed (No Upscale)",
}