import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import json
import asyncio
import aiohttp
from aiohttp import web
import io
import server
from concurrent.futures import Future

# --- Config Management ---
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "gpu_config.json")

def get_default_config():
    """Returns the default configuration dictionary. Single source of truth."""
    return {
        "master": {"port": 8188, "cuda_device": 0},
        "workers": [],
        "settings": {}
    }

def load_config():
    """Loads the config, falling back to defaults if the file is missing or invalid."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[MultiGPU Config] Error loading config, using defaults: {e}")
    return get_default_config()

def save_config(config):
    """Saves the configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"[MultiGPU Config] Error saving config: {e}")
        return False

def ensure_config_exists():
    """Creates default config file if it doesn't exist. Used by __init__.py"""
    if not os.path.exists(CONFIG_FILE):
        default_config = get_default_config()
        if save_config(default_config):
            print("[MultiGPU] Created default config file")
        else:
            print("[MultiGPU] Could not create default config file")

# --- Helper Functions ---
async def handle_api_error(request, error, status=500):
    """Standardized error response handler"""
    print(f"[MultiGPU] API Error: {error}")
    return web.json_response({"status": "error", "message": str(error)}, status=status)

# --- API Endpoints ---
@server.PromptServer.instance.routes.get("/multigpu/config")
async def get_config_endpoint(request):
    config = load_config()
    return web.json_response(config)

@server.PromptServer.instance.routes.post("/multigpu/config/update_worker")
async def update_worker_endpoint(request):
    try:
        data = await request.json()
        worker_id = data.get("worker_id")
        enabled = data.get("enabled", False)
        
        if worker_id is None:
            return await handle_api_error(request, "Missing worker_id", 400)
            
        config = load_config()
        worker_found = False
        
        for worker in config.get("workers", []):
            if worker["id"] == worker_id:
                worker["enabled"] = enabled
                worker_found = True
                break
                
        if not worker_found:
            return await handle_api_error(request, f"Worker {worker_id} not found", 404)
            
        if save_config(config):
            return web.json_response({"status": "success"})
        else:
            return await handle_api_error(request, "Failed to save config")
    except Exception as e:
        return await handle_api_error(request, e, 400)

@server.PromptServer.instance.routes.post("/multigpu/config/update_setting")
async def update_setting_endpoint(request):
    """Updates a specific key in the settings object."""
    try:
        data = await request.json()
        key = data.get("key")
        value = data.get("value")

        if not key or value is None:
            return await handle_api_error(request, "Missing 'key' or 'value' in request", 400)

        config = load_config()
        if 'settings' not in config:
            config['settings'] = {}
        
        config['settings'][key] = value

        if save_config(config):
            return web.json_response({"status": "success", "message": f"Setting '{key}' updated."})
        else:
            return await handle_api_error(request, "Failed to save config")
    except Exception as e:
        return await handle_api_error(request, e, 400)


@server.PromptServer.instance.routes.post("/multigpu/prepare_job")
async def prepare_job_endpoint(request):
    try:
        data = await request.json()
        multi_job_id = data.get('multi_job_id')
        if not multi_job_id:
            return await handle_api_error(request, "Missing multi_job_id", 400)

        async with PENDING_JOBS_LOCK:
            if multi_job_id not in PENDING_JOBS:
                PENDING_JOBS[multi_job_id] = asyncio.Queue()
        
        print(f"[MultiGPU] Prepared queue for job {multi_job_id}")
        return web.json_response({"status": "success"})
    except Exception as e:
        return await handle_api_error(request, e)

@server.PromptServer.instance.routes.post("/multigpu/clear_memory")
async def clear_memory_endpoint(request):
    print("[MultiGPU] Received request to clear VRAM.")
    try:
        # Use ComfyUI's prompt server queue system like the /free endpoint does
        if hasattr(server.PromptServer.instance, 'prompt_queue'):
            server.PromptServer.instance.prompt_queue.set_flag("unload_models", True)
            server.PromptServer.instance.prompt_queue.set_flag("free_memory", True)
            print("[MultiGPU] Set queue flags for memory clearing.")
        
        # Wait a bit for the queue to process
        await asyncio.sleep(0.5)
        
        # Also do direct cleanup as backup, but with error handling
        import gc
        import comfy.model_management as mm
        
        try:
            mm.unload_all_models()
        except AttributeError as e:
            print(f"[MultiGPU] Warning during model unload: {e}")
        
        try:
            mm.soft_empty_cache()
        except Exception as e:
            print(f"[MultiGPU] Warning during cache clear: {e}")
        
        for _ in range(3):
            gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        print("[MultiGPU] VRAM cleared successfully.")
        return web.json_response({"status": "success", "message": "GPU memory cleared."})
    except Exception as e:
        # Even if there's an error, try to do basic cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[MultiGPU] Partial VRAM clear completed with warning: {e}")
        return web.json_response({"status": "success", "message": "GPU memory cleared (with warnings)"})


# --- Global State & Callback Endpoint ---
PENDING_JOBS = {}
PENDING_JOBS_LOCK = asyncio.Lock()

@server.PromptServer.instance.routes.post("/multigpu/job_complete")
async def job_complete_endpoint(request):
    try:
        data = await request.post()
        multi_job_id = data.get('multi_job_id')
        image_file = data.get('image')

        if not all([multi_job_id, image_file]):
            return await handle_api_error(request, "Missing job_id or image data", 400)

        # Process image
        img_data = image_file.file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np)[None,]

        async with PENDING_JOBS_LOCK:
            if multi_job_id in PENDING_JOBS:
                await PENDING_JOBS[multi_job_id].put(tensor)
                print(f"[MultiGPU] Received result for job {multi_job_id}")
                return web.json_response({"status": "success"})
            else:
                return await handle_api_error(request, "Job not found or already complete", 404)
    except Exception as e:
        return await handle_api_error(request, e)

# --- Collector Node ---
class MultiGPUCollectorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "images": ("IMAGE",) },
            "hidden": {
                "multi_job_id": ("STRING", {"default": ""}),
                "is_worker": ("BOOLEAN", {"default": False}),
                "master_url": ("STRING", {"default": ""}),
                "enabled_worker_ids": ("STRING", {"default": "[]"}),
                # --- FIX [7 of 7]: Type Mismatch ---
                # Changed from FLOAT to INT for type safety and clarity.
                "worker_batch_size": ("INT", {"default": 1, "min": 1, "max": 1024}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image"
    
    def run(self, images, multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]", worker_batch_size=1):
        if not multi_job_id:
            return (images,)

        main_loop = server.PromptServer.instance.loop
        future = Future()

        async def run_and_set_future():
            try:
                result = await self.execute(images, multi_job_id, is_worker, master_url, enabled_worker_ids, worker_batch_size)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        asyncio.run_coroutine_threadsafe(run_and_set_future(), main_loop)
        return future.result()

    async def send_image_to_master(self, image_tensor, multi_job_id, master_url, image_index):
        """Helper method to send a single image to the master"""
        img_np = (image_tensor.cpu().numpy() * 255.).astype(np.uint8)
        img = Image.fromarray(img_np)
        byte_io = io.BytesIO()
        img.save(byte_io, format='PNG')
        byte_io.seek(0)
        
        data = aiohttp.FormData()
        data.add_field('multi_job_id', multi_job_id)
        data.add_field('image', byte_io, filename=f'image_{image_index}.png', content_type='image/png')

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{master_url}/multigpu/job_complete", data=data) as response:
                    response.raise_for_status()
        except Exception as e:
            print(f"[MultiGPU Worker] Failed to send image {image_index+1} to master: {e}")

    async def execute(self, images, multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]", worker_batch_size=1):
        if is_worker:
            # Worker mode: send images to master
            print(f"[MultiGPU Worker] Job {multi_job_id} complete. Sending {images.shape[0]} image(s) to master at {master_url}")
            
            tasks = [self.send_image_to_master(images[i], multi_job_id, master_url, i) for i in range(images.shape[0])]
            await asyncio.gather(*tasks)
            return (images,)
        else:
            # Master mode: collect images from workers
            num_workers = len(json.loads(enabled_worker_ids))
            if num_workers == 0:
                return (images,)
            
            images_on_cpu = images.cpu()
            total_images_to_collect = int(num_workers * worker_batch_size)
            print(f"[MultiGPU Master] Job {multi_job_id}: Waiting for {total_images_to_collect} image(s)...")
            
            all_tensors = [images_on_cpu]
            
            async with PENDING_JOBS_LOCK:
                if multi_job_id not in PENDING_JOBS:
                    PENDING_JOBS[multi_job_id] = asyncio.Queue()
                q = PENDING_JOBS[multi_job_id]
            
            try:
                # Collect all images with a potentially dynamic timeout
                for i in range(total_images_to_collect):
                    worker_tensor = await asyncio.wait_for(q.get(), timeout=300.0)
                    all_tensors.append(worker_tensor)
                    print(f"[MultiGPU Master] Collected image {i+1}/{total_images_to_collect}")
            except asyncio.TimeoutError:
                print(f"[MultiGPU Master] Timeout waiting for worker images (got {len(all_tensors)-1}/{total_images_to_collect})")
            finally:
                # Clean up job queue
                async with PENDING_JOBS_LOCK:
                    if multi_job_id in PENDING_JOBS:
                        del PENDING_JOBS[multi_job_id]

            combined = torch.cat(all_tensors, dim=0)
            print(f"[MultiGPU Master] Job {multi_job_id} complete. Combined {combined.shape[0]} images total.")
            return (combined,)

NODE_CLASS_MAPPINGS = { "MultiGPUCollector": MultiGPUCollectorNode }
NODE_DISPLAY_NAME_MAPPINGS = { "MultiGPUCollector": "Multi-GPU Collector" }
