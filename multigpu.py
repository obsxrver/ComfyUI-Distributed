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
        worker_id = data.get('worker_id')
        image_index = data.get('image_index')
        is_last = data.get('is_last', 'False').lower() == 'true'

        if not all([multi_job_id, image_file]):
            return await handle_api_error(request, "Missing job_id or image data", 400)

        # Process image
        img_data = image_file.file.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np)[None,]

        async with PENDING_JOBS_LOCK:
            if multi_job_id in PENDING_JOBS:
                await PENDING_JOBS[multi_job_id].put({
                    'tensor': tensor,
                    'worker_id': worker_id,
                    'image_index': int(image_index) if image_index else 0,
                    'is_last': is_last
                })
                print(f"[MultiGPU] Received result for job {multi_job_id} from worker {worker_id} (last: {is_last})")
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
                "worker_batch_size": ("INT", {"default": 1, "min": 1, "max": 1024}),
                "worker_id": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "image"
    
    def run(self, images, multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]", worker_batch_size=1, worker_id=""):
        if not multi_job_id:
            return (images,)

        main_loop = server.PromptServer.instance.loop
        future = Future()

        async def run_and_set_future():
            try:
                result = await self.execute(images, multi_job_id, is_worker, master_url, enabled_worker_ids, worker_batch_size, worker_id)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        asyncio.run_coroutine_threadsafe(run_and_set_future(), main_loop)
        return future.result()

    async def send_image_to_master(self, image_tensor, multi_job_id, master_url, image_index, worker_id, is_last=False):
        """Helper method to send a single image to the master"""
        img_np = (image_tensor.cpu().numpy() * 255.).astype(np.uint8)
        img = Image.fromarray(img_np)
        byte_io = io.BytesIO()
        img.save(byte_io, format='PNG')
        byte_io.seek(0)
        
        data = aiohttp.FormData()
        data.add_field('multi_job_id', multi_job_id)
        data.add_field('worker_id', str(worker_id))
        data.add_field('image_index', str(image_index))
        data.add_field('is_last', str(is_last))
        data.add_field('image', byte_io, filename=f'image_{image_index}.png', content_type='image/png')

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{master_url}/multigpu/job_complete", data=data) as response:
                    response.raise_for_status()
        except Exception as e:
            print(f"[MultiGPU Worker] Failed to send image {image_index+1} to master: {e}")

    async def execute(self, images, multi_job_id="", is_worker=False, master_url="", enabled_worker_ids="[]", worker_batch_size=1, worker_id=""):
        if is_worker:
            # Worker mode: send images to master
            print(f"[MultiGPU Worker] Job {multi_job_id} complete. Sending {images.shape[0]} image(s) to master")
            
            # Send all images, marking the last one
            for i in range(images.shape[0]):
                is_last = (i == images.shape[0] - 1)
                await self.send_image_to_master(images[i], multi_job_id, master_url, i, worker_id, is_last)
            
            return (images,)
        else:
            # Master mode: collect images from workers
            enabled_workers = json.loads(enabled_worker_ids)
            num_workers = len(enabled_workers)
            if num_workers == 0:
                return (images,)
            
            images_on_cpu = images.cpu()
            print(f"[MultiGPU Master] Job {multi_job_id}: Collecting images from {num_workers} workers...")
            
            # Store master images first
            master_batch_size = images.shape[0]
            
            # Initialize storage for collected images
            worker_images = {}  # Dict to store images by worker_id and index
            
            async with PENDING_JOBS_LOCK:
                if multi_job_id not in PENDING_JOBS:
                    PENDING_JOBS[multi_job_id] = asyncio.Queue()
                q = PENDING_JOBS[multi_job_id]
            
            # Collect images until all workers report they're done
            collected_count = 0
            workers_done = set()
            
            # Use a reasonable timeout for the first image
            timeout = 10.0
            
            while len(workers_done) < num_workers:
                try:
                    result = await asyncio.wait_for(q.get(), timeout=timeout)
                    worker_id = result['worker_id']
                    image_index = result['image_index']
                    tensor = result['tensor']
                    is_last = result.get('is_last', False)
                    
                    if worker_id not in worker_images:
                        worker_images[worker_id] = {}
                    worker_images[worker_id][image_index] = tensor
                    
                    collected_count += 1
                    
                    # Once we start receiving images, use shorter timeout
                    timeout = 10.0
                    
                    if is_last:
                        workers_done.add(worker_id)
                        print(f"[MultiGPU Master] Worker {worker_id} done. Collected {len(worker_images[worker_id])} images")
                    else:
                        print(f"[MultiGPU Master] Collected image {image_index + 1} from worker {worker_id}")
                    
                except asyncio.TimeoutError:
                    missing_workers = set(enabled_workers) - workers_done
                    print(f"[MultiGPU Master] Timeout. Still waiting for workers: {list(missing_workers)}")
                    break
            
            total_collected = sum(len(imgs) for imgs in worker_images.values())
            print(f"[MultiGPU Master] Collection complete. Received {total_collected} images from {len(workers_done)} workers")
            
            # Clean up job queue
            async with PENDING_JOBS_LOCK:
                if multi_job_id in PENDING_JOBS:
                    del PENDING_JOBS[multi_job_id]

            # Reorder images according to seed distribution pattern
            # Pattern: master img 1, master img 2, worker 1 img 1, worker 1 img 2, worker 2 img 1, worker 2 img 2, etc.
            ordered_tensors = []
            
            # Add master images first
            for i in range(master_batch_size):
                ordered_tensors.append(images_on_cpu[i:i+1])
            
            # Add worker images in order
            for worker_id in enabled_workers:
                # Convert worker_id to string since it comes as string from form data
                worker_id_str = str(worker_id)
                if worker_id_str in worker_images:
                    # Sort by image index for each worker
                    for idx in sorted(worker_images[worker_id_str].keys()):
                        ordered_tensors.append(worker_images[worker_id_str][idx])
            
            combined = torch.cat(ordered_tensors, dim=0)
            print(f"[MultiGPU Master] Job {multi_job_id} complete. Combined {combined.shape[0]} images total.")
            return (combined,)

# --- Distributor Node ---
class MultiGPUDistributor:
    """
    Distributes seed values across multiple GPUs.
    On master: passes through the original seed.
    On workers: adds offset based on worker ID.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "max": 1125899906842624,
                    "forceInput": False  # Widget by default, can be converted to input
                }),
            },
            "hidden": {
                "is_worker": ("BOOLEAN", {"default": False}),
                "worker_id": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "distribute"
    CATEGORY = "utils"
    
    def distribute(self, seed, is_worker=False, worker_id=""):
        if not is_worker:
            # Master node: pass through original values
            print(f"[MultiGPU Distributor] Master: seed={seed}")
            return (seed,)
        else:
            # Worker node: apply offset based on worker index
            # Find worker index from enabled_worker_ids
            try:
                # Worker IDs are passed as "worker_0", "worker_1", etc.
                if worker_id.startswith("worker_"):
                    worker_index = int(worker_id.split("_")[1])
                else:
                    # Fallback: try to parse as direct index
                    worker_index = int(worker_id)
                
                offset = worker_index + 1
                new_seed = seed + offset
                print(f"[MultiGPU Distributor] Worker {worker_index}: seed={seed} → {new_seed}")
                return (new_seed,)
            except (ValueError, IndexError) as e:
                print(f"[MultiGPU Distributor] Error parsing worker_id '{worker_id}': {e}")
                # Fallback: return original seed
                return (seed,)

NODE_CLASS_MAPPINGS = { 
    "MultiGPUCollector": MultiGPUCollectorNode,
    "MultiGPUDistributor": MultiGPUDistributor
}
NODE_DISPLAY_NAME_MAPPINGS = { 
    "MultiGPUCollector": "Multi-GPU Collector",
    "MultiGPUDistributor": "Multi-GPU Seed"
}
