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
- **Web UI Integration**: Seamless control panel integrated into ComfyUI's interface
- **Smart Launch**: Launcher checks for running instances and starts only those that aren't running

## How It Works

1. Launch multiple ComfyUI instances, each assigned to a different GPU
2. One instance acts as the master (typically on CUDA device 0)
3. Add the "Multi-GPU Collector" node to your workflow
4. The extension automatically distributes batches across available GPUs
5. Results are collected and combined by the master instance

For example, with 4 GPUs and a batch size of 2, you'll generate 8 images in parallel.

## Requirements

- ComfyUI installation
- Multiple NVIDIA GPUs
- Python 3.8+
- Windows (for the provided launcher script) or Linux/Mac with manual setup

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-Distributed.git
   ```

2. No additional dependencies required - uses ComfyUI's existing libraries

## Usage

### Quick Start (Windows)

1. Copy the `launch_distributed.bat` file to your ComfyUI root directory (alongside other .bat files like `run_nvidia_gpu.bat`)

2. Edit the `gpu_config.json` file in `ComfyUI/custom_nodes/ComfyUI-Distributed/` to match your GPU setup:
   - Adjust the number of workers based on your available GPUs
   - Update CUDA device numbers if needed
   - Set appropriate ports (ensure they're not in use)
   - Add custom launch arguments in `extra_args` for master and workers

3. Run the launcher script from the ComfyUI root directory:
   ```bash
   launch_distributed.bat
   ```

4. Open the master instance (usually http://localhost:8188)

5. Add the "Multi-GPU Collector" node to your workflow

6. Use the GPU control panel in the sidebar to:
   - View GPU status and worker information
   - Monitor active instances

### Manual Setup (All Platforms)

Start each ComfyUI instance with the appropriate CUDA device:

```bash
# Master instance (GPU 0)
CUDA_VISIBLE_DEVICES=0 python main.py --port 8188

# Worker instance (GPU 1)
CUDA_VISIBLE_DEVICES=1 python main.py --port 8189 --enable-cors-header

# Worker instance (GPU 2)
CUDA_VISIBLE_DEVICES=2 python main.py --port 8190 --enable-cors-header

# Continue for additional GPUs...
```

### Configuration

The system uses a `gpu_config.json` file for configuration. Default configuration:

```json
{
  "master": {
    "port": 8480,
    "cuda_device": 0,
    "extra_args": "--mmap-torch-files --highvram --disable-smart-memory"
  },
  "workers": [
    {
      "id": 1,
      "name": "GPU 1",
      "cuda_device": 1,
      "port": 8180,
      "enabled": true,
      "extra_args": "--mmap-torch-files --highvram --disable-smart-memory"
    },
    {
      "id": 2,
      "name": "GPU 2",
      "cuda_device": 2,
      "port": 8280,
      "enabled": true,
      "extra_args": "--mmap-torch-files --highvram --disable-smart-memory"
    }
  ],
  "settings": {
    "retry_delay_ms": 500
  }
}
```

**Configuration Options:**
- `extra_args`: Custom launch arguments for each instance (e.g., memory management flags)
- `retry_delay_ms`: Delay between connection attempts during startup
- Worker `enabled` flag: Control which GPUs are active

## Workflow Integration

The Multi-GPU Collector node integrates seamlessly with existing ComfyUI workflows:

1. Connect the node after your image generation pipeline
2. The node will automatically detect batch sizes and distribute work
3. All GPUs will process their assigned batches
4. Results are collected and passed to the next node as a single batch

### Important Node Requirements

- **Batch Node**: Your workflow must include a node with the title "BATCH" or "BATCH SIZE" (case-insensitive). This is required for the Multi-GPU system to detect the batch size for distribution.
- The system will show an error if no batch node with the correct title is found.

## Performance Considerations

- **Parallel Execution**: All enabled GPUs process simultaneously for maximum throughput
- **Batch Size**: Total generation = batch_size × number_of_active_GPUs
- **Smart Launching**: The launcher only starts instances that aren't already running
- **Memory Management**: Use appropriate `extra_args` in configuration for memory optimization

## Troubleshooting

- **Workers not connecting**: Ensure CORS headers are enabled (`--enable-cors-header`)
- **VRAM errors**: Add memory management flags in `extra_args` (e.g., `--highvram`, `--disable-smart-memory`)
- **Port conflicts**: Ensure each instance uses a unique port
- **Partial startup**: The launcher will automatically start only missing instances without affecting running ones

## Development

This project is under active development. Contributions are welcome!

### Planned Features/Improvements

- Support for remote workers over network
   - Worker status (read/in progress/offline)
   - Restart worker button
- ~Better VRAM management for multi-GPU setups~
- Better Seed handling
- Better Batch handling
- Handle worker failure
- Compatibility with FLUX Continuum `CTRL+SHIFT+C` shortcut

## License

Same as ComfyUI - see ComfyUI repository for details.

## Acknowledgments

Built as an extension for the amazing [ComfyUI](https://github.com/comfyanonymous/ComfyUI) project.
