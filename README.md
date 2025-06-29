# ComfyUI-Distributed

> ⚠️ **DEVELOPMENT STATUS**: This project is currently in active development and is not yet production-ready. Features may change, and bugs may exist. Use at your own risk.

A custom node extension for ComfyUI that enables distributed image generation across multiple GPUs through a master-worker architecture.

![image](https://github.com/user-attachments/assets/1e95e9e6-5237-4b24-8500-ac3b2703a1ea)

## Overview

ComfyUI-Distributed extends ComfyUI with the ability to distribute batch processing across multiple GPUs, allowing for linear scaling of image generation performance. It achieves this by coordinating multiple ComfyUI instances, each running on a different GPU, with one instance acting as the master coordinator.

## Features

- **Multi-GPU Distributed Processing**: Leverage all available GPUs simultaneously
- **Master-Worker Architecture**: Automatic coordination between GPU instances
- **Dynamic Worker Management**: Enable/disable GPUs on-the-fly through the UI
- **Parallel Execution**: All enabled GPUs process batches simultaneously for maximum speed
- **Automatic Seed Distribution**: Smart seed offsetting across workers
- **Web UI Integration**: Seamless control panel integrated into ComfyUI's sidebar
- **Smart Launcher**: Unified launcher with multiple modes and instance detection
- **VRAM Management**: Clear worker VRAM remotely through the UI
- **Flexible Configuration**: JSON-based configuration with runtime updates

## How It Works

1. Launch multiple ComfyUI instances using the unified launcher
2. One instance acts as the master (typically on CUDA device 0)
3. Add the "Multi-GPU Collector" node to your workflow
4. Optionally add "Multi-GPU Distributor" nodes for automatic seed distribution
5. The extension automatically distributes batches across available GPUs
6. Results are collected and combined by the master instance

For example, with 4 GPUs and a batch size of 2, you'll generate 8 images in parallel.

## Nodes

### Multi-GPU Distributor
Handles automatic seed distribution for diverse results:
- Master uses the original seed value
- Workers automatically get offset seeds (seed + worker_index + 1)
- Can connect to any node that accepts seed/batch_size inputs
- Ensures each GPU generates unique images

### Multi-GPU Collector
The core node that handles job result collection:
- Automatically detects enabled workers
- Collects and combines results in a predictable order

## Requirements

- Virtual environment with ComfyUI installed
- ComfyUI installation with ComfyUI CLI (`comfy` command)
- Multiple NVIDIA GPUs
- Python 3.8+

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-Distributed.git
   ```

2. No additional dependencies required - uses ComfyUI's existing libraries

## Usage

### Quick Start (Windows)

1. Copy the `launch_distributed.bat` file to your ComfyUI root directory (alongside your `venv` folder)

2. Edit the `gpu_config.json` file in `ComfyUI/custom_nodes/ComfyUI-Distributed/` to match your GPU setup:
   - Adjust the number of workers based on your available GPUs
   - Update CUDA device numbers if needed
   - Set appropriate ports (ensure they're not in use)
   - Add custom launch arguments in `extra_args`

3. Run the launcher from the ComfyUI root directory:
   ```bash
   launch_distributed.bat
   ```
   
   Or use command-line options:
   ```bash
   launch_distributed.bat master      # Launch master only
   launch_distributed.bat all-fast    # Launch all with 2-second delays
   launch_distributed.bat all-slow    # Launch all, wait for each to be ready
   launch_distributed.bat venv        # Open command prompt with venv activated
   launch_distributed.bat update      # Update ComfyUI
   launch_distributed.bat update-nodes # Update all custom nodes
   ```

4. Open the master instance (check the launcher output for the URL)

5. Add the "Multi-GPU Collector" node to your workflow

6. Use the Multi-GPU sidebar panel to:
   - Enable/disable worker GPUs
   - View worker status
   - Clear worker VRAM

### Configuration

The system uses a `gpu_config.json` file for configuration. Example configuration:

```json
{
  "master": {
    "port": 8080,
    "cuda_device": 0,
    "extra_args": "--mmap-torch-files --listen"
  },
  "workers": [
    {
      "id": 1,
      "name": "GPU 1",
      "cuda_device": 1,
      "port": 8180,
      "enabled": true,
      "extra_args": "--mmap-torch-files --listen"
    },
    {
      "id": 2,
      "name": "GPU 2",
      "cuda_device": 2,
      "port": 8280,
      "enabled": true,
      "extra_args": "--mmap-torch-files --listen"
    },
    {
      "id": 3,
      "name": "GPU 3",
      "cuda_device": 3,
      "port": 8380,
      "enabled": false,
      "extra_args": "--mmap-torch-files --listen --reserve-vram 2"
    }
  ],
  "settings": {}
}
```

**Configuration Options:**
- `master`: Master instance configuration
  - `port`: Port number for the master instance
  - `cuda_device`: CUDA device index
  - `extra_args`: Custom launch arguments
- `workers`: Array of worker configurations
  - `id`: Unique worker identifier
  - `name`: Display name in UI
  - `cuda_device`: CUDA device index
  - `port`: Port number for the worker
  - `enabled`: Whether the worker is active
  - `extra_args`: Custom launch arguments
  - `host` (optional): Remote host address
- `settings`: Additional settings (reserved for future use)

### Remote Workers

Remote workers are fully supported! Add a `host` field to any worker configuration:

```json
{
  "id": 3,
  "name": "Remote GPU",
  "host": "192.168.1.100",
  "port": 8190,
  "enabled": true,
  "extra_args": "--enable-cors-header --listen"
}
```

Ensure remote workers:
- Are started with `--listen --enable-cors-header`
- Are accessible from the master instance
- Have the same models and custom nodes installed

## Workflow Integration

### Basic Setup
1. Add "Multi-GPU Collector" node after your image generation pipeline
2. The node automatically handles distribution and collection
3. Connect subsequent nodes normally - they'll receive the combined batch

### Seed Distribution
This node is optional, it will help you control seeds across workers. Without it, you will generate exactly the same workflow as the master (same seed):
1. Add "Multi-GPU Seed" node(s) before your samplers
2. Connect the seed output to your sampling node(s)
3. Each GPU will automatically use different seeds:
   - Master: original seed
   - Worker 0: seed + 1
   - Worker 1: seed + 2
   - etc.

### Result Ordering
Images are returned in a predictable order:
1. All images from master (in batch order)
2. All images from worker 1 (in batch order)
3. All images from worker 2 (in batch order)
4. etc.

## API Endpoints

The extension provides REST API endpoints for integration:

- `GET /multigpu/config` - Retrieve current configuration
- `POST /multigpu/config/update_worker` - Enable/disable a worker
- `POST /multigpu/config/update_setting` - Update configuration settings
- `POST /multigpu/prepare_job` - Prepare a multi-GPU job
- `POST /multigpu/clear_memory` - Clear VRAM on workers
- `POST /multigpu/job_complete` - Worker callback for job completion

## Performance Considerations

- **Parallel Execution**: All enabled GPUs process simultaneously for maximum throughput
- **Batch Size**: Total generation = batch_size × (1 master + number_of_enabled_workers)
- **Smart Launching**: The launcher only starts instances that aren't already running
- **Memory Management**: Use appropriate `extra_args` in configuration for memory optimization
- **Network Overhead**: Minimal - only final images are transferred between instances

## Troubleshooting

- **Workers not connecting**: Ensure CORS headers are enabled (`--enable-cors-header`)
- **VRAM errors**: Add memory management flags in `extra_args` (e.g., `--highvram`, `--disable-smart-memory`)
- **Port conflicts**: Ensure each instance uses a unique port
- **Launcher errors**: Ensure the batch file is in the ComfyUI root directory with the `venv` folder
- **Batch node not found**: Ensure your workflow has a node titled "BATCH" or "BATCH SIZE"
- **ComfyUI CLI not found**: Make sure ComfyUI CLI is installed (`pip install comfyui-cli`)

## Development

This project is under active development. Contributions are welcome!

### Planned Features/Improvements

- ~Custom launcher~
   - ~Launch local workers~
   - Launch remote workers (via SSH)
- ~Support for remote workers over network~  
   - Worker status (ready/in progress/offline)
   - Restart worker button
- ~Better VRAM management for multi-GPU setups~
- ~Better Seed handling~
- Handle worker failure
- Handle multiple image output nodes within 1 workflow
- ~Compatibility with FLUX Continuum `CTRL+SHIFT+C` shortcut~ [Commit 027629c](https://github.com/robertvoy/ComfyUI-Flux-Continuum/commit/027629c753dd3aae1ceeff5214ceb701943dd3fe)

## License

Same as ComfyUI - see ComfyUI repository for details.

## Acknowledgments

Built as an extension for the amazing [ComfyUI](https://github.com/comfyanonymous/ComfyUI) project.
