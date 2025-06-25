# ComfyUI-Distributed

> ⚠️ **DEVELOPMENT STATUS**: This project is currently in active development and is not yet production-ready. Features may change, and bugs may exist. Use at your own risk.

A custom node extension for ComfyUI that enables distributed image generation across multiple GPUs through a master-worker architecture.

## Overview

ComfyUI-Distributed extends ComfyUI with the ability to distribute batch processing across multiple GPUs, allowing for linear scaling of image generation performance. It achieves this by coordinating multiple ComfyUI instances, each running on a different GPU, with one instance acting as the master coordinator.

## Features

- **Multi-GPU Distributed Processing**: Leverage all available GPUs simultaneously
- **Master-Worker Architecture**: Automatic coordination between GPU instances
- **Dynamic Worker Management**: Enable/disable GPUs on-the-fly through the UI
- **Execution Modes**:
  - **Parallel Mode**: All GPUs process simultaneously for maximum speed
  - **Staggered Mode**: Sequential GPU startup with configurable delays to manage VRAM usage
- **VRAM Management**: Built-in tools to clear GPU memory across all instances
- **Web UI Integration**: Seamless control panel integrated into ComfyUI's interface

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

2. Run the launcher script from the ComfyUI root directory:
   ```bash
   launch_distributed.bat
   ```
   This will automatically start ComfyUI instances for each GPU

3. Open the master instance (usually http://localhost:8188)

4. Add the "Multi-GPU Collector" node to your workflow

5. Use the GPU control panel in the sidebar to:
   - Select which GPUs to use
   - Choose between parallel or staggered execution
   - Clear VRAM when needed

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
    "port": 8188,
    "device": 0,
    "enabled": true
  },
  "workers": [
    {
      "port": 8189,
      "device": 1,
      "enabled": true,
      "extra_args": "--enable-cors-header"
    },
    {
      "port": 8190,
      "device": 2,
      "enabled": true,
      "extra_args": "--enable-cors-header"
    }
  ],
  "settings": {
    "stagger_delay": 2.0,
    "job_timeout": 120
  }
}
```

## Workflow Integration

The Multi-GPU Collector node integrates seamlessly with existing ComfyUI workflows:

1. Connect the node after your image generation pipeline
2. The node will automatically detect batch sizes and distribute work
3. All GPUs will process their assigned batches
4. Results are collected and passed to the next node as a single batch

## Performance Considerations

- **Parallel Mode**: Best for workflows with consistent VRAM usage
- **Staggered Mode**: Ideal for memory-intensive workflows or when GPUs have different VRAM capacities
- **Batch Size**: Total generation = batch_size × number_of_active_GPUs

## Troubleshooting

- **Workers not connecting**: Ensure CORS headers are enabled (`--enable-cors-header`)
- **VRAM errors**: Use staggered mode with appropriate delays
- **Timeout errors**: Increase `job_timeout` in configuration
- **Port conflicts**: Ensure each instance uses a unique port

## Development

This project is under active development. Contributions are welcome!

### Known Issues

- Windows Terminal required for the launcher script
- Manual configuration needed for more than 4 GPUs
- Some custom nodes may not be compatible with distributed processing

### Planned Features

- Linux/Mac launcher scripts
- Automatic GPU detection and configuration
- Load balancing based on GPU capabilities
- Support for remote workers over network
- Queue management improvements

## License

Same as ComfyUI - see ComfyUI repository for details.

## Acknowledgments

Built as an extension for the amazing [ComfyUI](https://github.com/comfyanonymous/ComfyUI) project.