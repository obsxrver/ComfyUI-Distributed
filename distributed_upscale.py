import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import json
import asyncio
import aiohttp
from aiohttp import web
import io
import server
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

# Helper function to ensure persistent state is initialized
def ensure_tile_jobs_initialized():
    """Ensure tile job storage is initialized on the server instance."""
    prompt_server = server.PromptServer.instance
    if not hasattr(prompt_server, 'distributed_pending_tile_jobs'):
        debug_log("Initializing persistent tile job queue on server instance.")
        prompt_server.distributed_pending_tile_jobs = {}
        prompt_server.distributed_tile_jobs_lock = asyncio.Lock()
    return prompt_server

# Note: tensor_to_pil and pil_to_tensor are imported from utils.image

class UltimateSDUpscaleDistributed:
    """
    Distributed version of Ultimate SD Upscale (No Upscale).
    Distributes tile processing across multiple workers.
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
            },
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                "worker_id": ("STRING", {"default": ""}),
                "tile_indices": ("STRING", {"default": ""}),  # Unused - kept for compatibility
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image/upscaling"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution and ensure queues exist for the job."""
        multi_job_id = kwargs.get('multi_job_id', '')
        if multi_job_id:
            # Initialize queue for this job immediately
            prompt_server = ensure_tile_jobs_initialized()
            loop = get_server_loop()
            
            async def init_queue():
                async with prompt_server.distributed_tile_jobs_lock:
                    if multi_job_id not in prompt_server.distributed_pending_tile_jobs:
                        prompt_server.distributed_pending_tile_jobs[multi_job_id] = asyncio.Queue()
                        debug_log(f"UltimateSDUpscaleDistributed - Pre-initialized queue for job {multi_job_id}")
            
            try:
                asyncio.run_coroutine_threadsafe(init_queue(), loop).result(timeout=1.0)
            except Exception as e:
                debug_log(f"UltimateSDUpscaleDistributed - Error pre-initializing queue: {e}")
        
        return float("nan")  # Always re-execute
    
    def run(self, upscaled_image, model, positive, negative, vae, seed, steps, cfg, 
            sampler_name, scheduler, denoise, tile_width, tile_height, padding, 
            mask_blur, force_uniform_tiles, multi_job_id="", is_worker=False, 
            master_url="", enabled_worker_ids="[]", worker_id="", tile_indices=""):
        """Entry point - runs SYNCHRONOUSLY like Ultimate SD Upscaler."""
        if not multi_job_id:
            # No distributed processing, run single GPU version
            return self.process_single_gpu(upscaled_image, model, positive, negative, vae,
                                          seed, steps, cfg, sampler_name, scheduler, denoise,
                                          tile_width, tile_height, padding, mask_blur, force_uniform_tiles)
        
        if is_worker:
            # Worker mode: process tiles synchronously
            return self.process_worker_tiles(upscaled_image, model, positive, negative, vae,
                                           seed, steps, cfg, sampler_name, scheduler, denoise,
                                           tile_width, tile_height, padding, mask_blur,
                                           force_uniform_tiles, multi_job_id, master_url,
                                           worker_id, enabled_worker_ids)
        else:
            # Master mode: distribute and collect synchronously
            return self.process_master(upscaled_image, model, positive, negative, vae,
                                     seed, steps, cfg, sampler_name, scheduler, denoise,
                                     tile_width, tile_height, padding, mask_blur,
                                     force_uniform_tiles, multi_job_id, enabled_worker_ids)
    
    def process_worker_tiles(self, upscaled_image, model, positive, negative, vae,
                           seed, steps, cfg, sampler_name, scheduler, denoise,
                           tile_width, tile_height, padding, mask_blur,
                           force_uniform_tiles, multi_job_id, master_url,
                           worker_id, enabled_worker_ids):
        """Worker processing - SYNCHRONOUS tile processing with async communication."""
        # Round tile dimensions
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)
        
        # Calculate all tile positions
        _, height, width, _ = upscaled_image.shape
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height)
        
        # Get assigned tiles for this worker
        assigned_tiles = self._get_worker_tiles(all_tiles, enabled_worker_ids, worker_id)
        if not assigned_tiles:
            return (upscaled_image,)
        
        debug_log(f"UltimateSDUpscale Worker - Worker {worker_id} processing {len(assigned_tiles)} tiles")
        
        # Process tiles SYNCHRONOUSLY
        processed_tiles = []
        for tile_idx in assigned_tiles:
            x, y = all_tiles[tile_idx]
            
            # Extract tile
            tile_tensor, x1, y1, ew, eh = self.extract_tile_with_padding(
                upscaled_image, x, y, tile_width, tile_height, padding
            )
            
            # Process tile through SD (SYNCHRONOUS - no async!)
            processed_tile = self.process_tile(tile_tensor, model, positive, negative, vae,
                                             seed + tile_idx, steps, cfg, sampler_name, 
                                             scheduler, denoise)
            
            processed_tiles.append({
                'tile': processed_tile,
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
                      force_uniform_tiles, multi_job_id, enabled_worker_ids):
        """Master processing - SYNCHRONOUS with async collection."""
        # Round tile dimensions
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)
        
        # Get image dimensions
        _, height, width, _ = upscaled_image.shape
        
        # Calculate all tiles
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height)
        total_tiles = len(all_tiles)
        
        debug_log(f"UltimateSDUpscale Master - Total tiles: {total_tiles}")
        
        # Parse enabled workers
        enabled_workers = json.loads(enabled_worker_ids)
        num_workers = len(enabled_workers)
        
        if num_workers == 0:
            # No workers, process all tiles locally
            return self.process_single_gpu(upscaled_image, model, positive, negative, vae,
                                         seed, steps, cfg, sampler_name, scheduler, denoise,
                                         tile_width, tile_height, padding, mask_blur, force_uniform_tiles)
        
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
            # Try synchronous initialization as fallback
            prompt_server = ensure_tile_jobs_initialized()
            if multi_job_id not in prompt_server.distributed_pending_tile_jobs:
                prompt_server.distributed_pending_tile_jobs[multi_job_id] = asyncio.Queue()
                debug_log(f"UltimateSDUpscale Master - Queue initialized via fallback")
        
        # Don't try to prepare via API since we're using local queues
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        
        # Prepare result image
        image_pil = tensor_to_pil(upscaled_image, 0)
        result_image = image_pil.copy()
        
        # Mask will be created per tile for proper blending
        
        # Calculate master's tiles
        master_tiles_indices = self._get_master_tiles(all_tiles, num_workers)
        master_tiles = len(master_tiles_indices)
        
        debug_log(f"UltimateSDUpscale Master - Processing {master_tiles} tiles locally")
        
        # Process master tiles SYNCHRONOUSLY
        for tile_idx in master_tiles_indices:
            result_image = self._process_and_blend_tile(
                tile_idx, all_tiles[tile_idx], upscaled_image, result_image,
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise, tile_width, tile_height,
                padding, mask_blur, width, height
            )
        
        # Collect worker tiles using async
        worker_tiles_expected = total_tiles - master_tiles
        if worker_tiles_expected > 0:
            # Calculate tile distribution for active worker calculation
            tiles_per_participant = total_tiles // (num_workers + 1)
            remainder = total_tiles % (num_workers + 1)
            
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
                
                if worker_start_idx < total_tiles and worker_tile_count > 0:
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
                    
                    # Convert and blend
                    tile_pil = tensor_to_pil(tile_tensor, 0)
                    # Get the original tile position from the tile index
                    orig_x, orig_y = all_tiles[tile_idx]
                    # Create mask for this specific tile
                    tile_mask = self.create_tile_mask(width, height, orig_x, orig_y, tile_width, tile_height, mask_blur)
                    # Use extraction position and size for blending
                    result_image = self.blend_tile(result_image, tile_pil, 
                                                 x, y, (extracted_width, extracted_height), tile_mask, padding)
        
        # Convert back to tensor
        result_tensor = pil_to_tensor(result_image)
        if upscaled_image.is_cuda:
            result_tensor = result_tensor.cuda()
        
        debug_log(f"UltimateSDUpscale Master - Job {multi_job_id} complete")
        return (result_tensor,)
    
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
                               padding, mask_blur, image_width, image_height):
        """Process a single tile and blend it into the result image."""
        x, y = tile_pos
        
        # Extract and process tile
        tile_tensor, x1, y1, ew, eh = self.extract_tile_with_padding(
            upscaled_image, x, y, tile_width, tile_height, padding
        )
        
        processed_tile = self.process_tile(tile_tensor, model, positive, negative, vae,
                                         seed + tile_idx, steps, cfg, sampler_name, 
                                         scheduler, denoise)
        
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
    
    async def _init_job_queue(self, multi_job_id):
        """Initialize the job queue for collecting tiles."""
        prompt_server = ensure_tile_jobs_initialized()
        async with prompt_server.distributed_tile_jobs_lock:
            debug_log(f"UltimateSDUpscale Master - _init_job_queue: Checking if {multi_job_id} exists in distributed_pending_tile_jobs")
            debug_log(f"UltimateSDUpscale Master - _init_job_queue: Current jobs: {list(prompt_server.distributed_pending_tile_jobs.keys())}")
            if multi_job_id not in prompt_server.distributed_pending_tile_jobs:
                prompt_server.distributed_pending_tile_jobs[multi_job_id] = asyncio.Queue()
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
            q = prompt_server.distributed_pending_tile_jobs[multi_job_id]
        
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
                        tile_idx = tile_data['tile_idx']
                        # Store the full tile data including metadata
                        collected_tiles[tile_idx] = {
                            'tensor': tile_data['tensor'],
                            'tile_idx': tile_idx,
                            'x': tile_data['x'],
                            'y': tile_data['y'],
                            'extracted_width': tile_data['extracted_width'],
                            'extracted_height': tile_data['extracted_height'],
                            'padding': tile_data['padding'],
                            'worker_id': worker_id
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
    
    def round_to_multiple(self, value: int, multiple: int = 8) -> int:
        """Round value to nearest multiple."""
        return round(value / multiple) * multiple
    
    def calculate_tiles(self, image_width: int, image_height: int, 
                       tile_width: int, tile_height: int) -> List[Tuple[int, int]]:
        """Calculate tile positions for the image.
        
        Tiles are placed at grid positions without overlap in their placement.
        The overlap happens during extraction with padding."""
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
                    scheduler: str, denoise: float) -> torch.Tensor:
        """Process a single tile through SD sampling."""
        # Import here to avoid circular dependencies
        from nodes import common_ksampler, VAEEncode, VAEDecode
        
        # Convert to PIL and back to ensure clean tensor without gradient tracking
        tile_pil = tensor_to_pil(tile_tensor, 0)
        clean_tensor = pil_to_tensor(tile_pil)
        
        # Move to correct device
        if tile_tensor.is_cuda:
            clean_tensor = clean_tensor.cuda()
        
        # Encode to latent
        latent = VAEEncode().encode(vae, clean_tensor)[0]
        
        # Sample
        samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, 
                                positive, negative, latent, denoise=denoise)[0]
        
        # Decode back to image
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
        """Send all processed tiles to master in a single batch request."""
        data = aiohttp.FormData()
        data.add_field('multi_job_id', multi_job_id)
        data.add_field('worker_id', str(worker_id))
        data.add_field('is_last', 'True')  # Full batch, so last for this worker
        data.add_field('batch_size', str(len(processed_tiles)))
        data.add_field('padding', str(padding))
        
        # Add each tile as a separate field with metadata
        for i, tile_data in enumerate(processed_tiles):
            # Convert tensor to PIL
            img = tensor_to_pil(tile_data['tile'], 0)
            byte_io = io.BytesIO()
            img.save(byte_io, format='PNG', compress_level=0)
            byte_io.seek(0)
            
            # Add image field
            data.add_field(f'tile_{i}', byte_io, filename=f'tile_{i}.png', content_type='image/png')
            
            # Add metadata for this tile
            data.add_field(f'tile_{i}_idx', str(tile_data['tile_idx']))
            data.add_field(f'tile_{i}_x', str(tile_data['x']))
            data.add_field(f'tile_{i}_y', str(tile_data['y']))
            data.add_field(f'tile_{i}_width', str(tile_data['extracted_width']))
            data.add_field(f'tile_{i}_height', str(tile_data['extracted_height']))
        
        # Retry logic with exponential backoff
        max_retries = 5
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                session = await get_client_session()
                url = f"{master_url}/distributed/tile_complete"
                
                debug_log(f"UltimateSDUpscale Worker - Sending batch of {len(processed_tiles)} tiles to {url}, attempt {attempt + 1}")
                
                async with session.post(url, data=data) as response:
                    response.raise_for_status()
                    debug_log(f"UltimateSDUpscale Worker - Successfully sent batch of {len(processed_tiles)} tiles")
                    return
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    debug_log(f"UltimateSDUpscale Worker - Retry {attempt + 1}/{max_retries} after error: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    log(f"UltimateSDUpscale Worker - Failed to send batch after {max_retries} attempts: {e}")
                    raise

    async def send_tile_to_master(self, tile_tensor, multi_job_id, master_url, 
                                 tile_idx, x, y, extracted_width, extracted_height, 
                                 padding, worker_id, is_last=False):
        """Send processed tile to master with retry logic."""
        # Convert tensor to PIL image
        img = tensor_to_pil(tile_tensor, 0)
        
        # Store image bytes for retry
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG', compress_level=0)
        img_data = img_bytes.getvalue()
        
        # Retry logic with exponential backoff
        max_retries = 5
        retry_delay = 0.5  # Start with 500ms
        
        for attempt in range(max_retries):
            try:
                # Create fresh FormData for each attempt
                data = aiohttp.FormData()
                data.add_field('multi_job_id', multi_job_id)
                data.add_field('worker_id', str(worker_id))
                data.add_field('tile_idx', str(tile_idx))
                data.add_field('x', str(x))
                data.add_field('y', str(y))
                data.add_field('extracted_width', str(extracted_width))
                data.add_field('extracted_height', str(extracted_height))
                data.add_field('padding', str(padding))
                data.add_field('is_last', str(is_last))
                data.add_field('image', io.BytesIO(img_data), filename=f'tile_{tile_idx}.png', content_type='image/png')
                
                session = await get_client_session()
                timeout = aiohttp.ClientTimeout(total=TILE_TRANSFER_TIMEOUT)
                async with session.post(f"{master_url}/distributed/tile_complete", data=data, timeout=timeout) as response:
                    if response.status == 404 and attempt < max_retries - 1:
                        # Queue not ready yet, wait and retry
                        debug_log(f"UltimateSDUpscale Worker - Queue not ready for job {multi_job_id}, attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 5.0)  # Exponential backoff with cap at 5s
                        continue
                    response.raise_for_status()
                    debug_log(f"UltimateSDUpscale Worker - Successfully sent tile {tile_idx} on attempt {attempt + 1}")
                    return  # Success
            except aiohttp.ClientResponseError as e:
                if e.status == 404 and attempt < max_retries - 1:
                    debug_log(f"UltimateSDUpscale Worker - Got 404 for job {multi_job_id}, attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 5.0)
                    continue
                elif attempt == max_retries - 1:
                    log(f"UltimateSDUpscale Worker - Failed to send tile {tile_idx} after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                if attempt == max_retries - 1:
                    log(f"UltimateSDUpscale Worker - Failed to send tile {tile_idx} after {max_retries} attempts: {e}")
                    raise
                else:
                    debug_log(f"UltimateSDUpscale Worker - Error on attempt {attempt + 1}: {e}, retrying...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 5.0)
    
    
    def process_single_gpu(self, upscaled_image, model, positive, negative, vae,
                          seed, steps, cfg, sampler_name, scheduler, denoise,
                          tile_width, tile_height, padding, mask_blur, force_uniform_tiles):
        """Process all tiles on a single GPU (no distribution)."""
        # Round tile dimensions
        tile_width = self.round_to_multiple(tile_width)
        tile_height = self.round_to_multiple(tile_height)
        
        # Get image dimensions
        _, height, width, _ = upscaled_image.shape
        
        # Calculate all tiles
        all_tiles = self.calculate_tiles(width, height, tile_width, tile_height)
        
        debug_log(f"UltimateSDUpscale - Processing {len(all_tiles)} tiles locally")
        
        # Convert to PIL
        image_pil = tensor_to_pil(upscaled_image, 0)
        result_image = image_pil.copy()
        
        # Mask will be created per tile for proper blending
        
        # Process each tile
        for idx, tile_pos in enumerate(all_tiles):
            result_image = self._process_and_blend_tile(
                idx, tile_pos, upscaled_image, result_image,
                model, positive, negative, vae, seed, steps, cfg,
                sampler_name, scheduler, denoise, tile_width, tile_height,
                padding, mask_blur, width, height
            )
        
        # Convert back to tensor
        result_tensor = pil_to_tensor(result_image)
        if upscaled_image.is_cuda:
            result_tensor = result_tensor.cuda()
        
        return (result_tensor,)


# Ensure initialization before registering routes
ensure_tile_jobs_initialized()

# API Endpoint for tile completion
@server.PromptServer.instance.routes.post("/distributed/tile_complete")
async def tile_complete_endpoint(request):
    """Endpoint for receiving completed tiles from workers."""
    try:
        data = await request.post()
        multi_job_id = data.get('multi_job_id')
        worker_id = data.get('worker_id')
        is_last = data.get('is_last', 'False').lower() == 'true'
        
        if multi_job_id is None or worker_id is None:
            return await handle_api_error(request, "Missing multi_job_id or worker_id", 400)

        prompt_server = ensure_tile_jobs_initialized()
        
        # Check for batch mode
        batch_size = int(data.get('batch_size', 0))
        tiles = []
        
        if batch_size > 0:
            # Batch mode: Extract multiple tiles
            padding = int(data.get('padding', 32))
            debug_log(f"UltimateSDUpscale - tile_complete batch - job_id: {multi_job_id}, worker: {worker_id}, batch_size: {batch_size}")
            
            for i in range(batch_size):
                tile_field = data.get(f'tile_{i}')
                if tile_field is None:
                    return await handle_api_error(request, f"Missing tile_{i}", 400)
                
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
                    log(f"Error processing tile {i} from worker {worker_id}: {e}")
                    return await handle_api_error(request, f"Tile processing error: {e}", 400)
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
                    'padding': padding
                }]
            except Exception as e:
                log(f"Error processing tile from worker {worker_id}: {e}")
                return await handle_api_error(request, f"Tile processing error: {e}", 400)

        # Put tiles into queue
        async with prompt_server.distributed_tile_jobs_lock:
            debug_log(f"UltimateSDUpscale - tile_complete: Checking distributed_pending_tile_jobs for job {multi_job_id}")
            debug_log(f"UltimateSDUpscale - tile_complete: Current jobs in distributed_pending_tile_jobs: {list(prompt_server.distributed_pending_tile_jobs.keys())}")
            
            if multi_job_id in prompt_server.distributed_pending_tile_jobs:
                if batch_size > 0:
                    # Put batch as single item  
                    await prompt_server.distributed_pending_tile_jobs[multi_job_id].put({
                        'worker_id': worker_id,
                        'tiles': tiles,
                        'is_last': is_last
                    })
                    debug_log(f"UltimateSDUpscale - Received batch of {len(tiles)} tiles for job {multi_job_id} from worker {worker_id}")
                else:
                    # Put single tile (backward compat)
                    tile_data = tiles[0]
                    await prompt_server.distributed_pending_tile_jobs[multi_job_id].put({
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