<div align="center">
<img width="1680" height="889" alt="image" src="https://github.com/user-attachments/assets/8bbb992a-2c81-4674-a7a1-c093a6215ffc" />
<br><br>
<a href="">📺Video Tutorial</a> |
<a href="">📝Setup Guides</a> | 
<a href="https://github.com/robertvoy/ComfyUI-Distributed/tree/main/workflows">⚙️Workflows</a> |
<a href="https://buymeacoffee.com/robertvoy">🎁Donation</a> 
<br><br>
</div>

> **A powerful extension for ComfyUI that enables distributed and parallel processing across multiple GPUs and machines. Generate more images and videos and accelerate your upscaling workflows by leveraging all available GPU resources in your network.**

---

## Key Features

#### Parallel Workflow Processing
- Run the same workflow on multiple GPUs simultaneously with different seeds
- Get multiple outputs in the same time it takes to generate one
- The more workers you have, the more output you get
- Supports both video and images

#### Distributed Upscaling
- Supercharge Ultimate SD Upscale by having tiles processed across multiple GPUs
- Intelligent work distribution for asymmetrical GPU setups
- Static distribution for similar/same GPUs
- Supports both single images and image batches

#### Management & Monitoring
- **Worker Management** - Launch and monitor workers from the UI
- **Network Support** - Use GPUs across different machines on your network
- **Cloud Support** - Use GPUs from cloud providers
- **Real-time Monitoring** - Track worker status and performance from the UI
- **Easy Configuration** - JSON-based configuration with UI controls

---

## Requirements

- ComfyUI installation
   - Desktop app not supported currently
- Multiple GPUs
- That's it

---

## Installation

1. **Clone this repository** into your ComfyUI custom nodes directory:
   ```bash
   git clone https://github.com/robertvoy/ComfyUI-Distributed.git
   ```

2. **Restart ComfyUI** - If you'll be using remote/cloud workers, add `--enable-cors-header` to your launch arguments on the master

---



---

## Nodes

### Distributed Collector
> Collects and combines results from distributed processing

**Usage**: Place after any image generation node to enable distributed processing. Works automatically when workers are enabled.

### Distributed Seed
> Ensures unique seeds across distributed workers for varied generations

**Usage**: Connect to any seed input. Each worker automatically receives an offset seed to ensure randomisation. Alternatively, you can connect a seed node directly to this node's seed input, which will automatically handle seed offsetting across all workers.

### Ultimate SD Upscale Distributed
> Distributed version of Ultimate SD Upscale that processes tiles across multiple GPUs, making upscaling much faster

**Usage**:
1. Upscale your image with a regular upscale model (ESRGAN, etc.)
2. Feed the upscaled image into this node
3. Configure tile settings
4. Enable workers for fast processing

---

## UI Features

### Distributed GPU Panel

The control centre for your distributed setup:

| Feature | Description |
|---------|-------------|
| **Worker Status** | Real-time status indicators |
| **Launch/Stop** | Control individual local workers |
| **Clear Memory** | Free VRAM on all workers |
| **Interrupt** | Stop current processing |
| **Worker Logs** | View real-time logs |

---

## Workflow Examples

### Basic Distributed Generation

![Clipboard Image (1)](https://github.com/user-attachments/assets/e8e46d97-d698-4c18-b4e5-1e1a2f4f7da3)

1. Create your normal ComfyUI workflow
2. Add **Distributed Seed** → connect to sampler's seed
3. Add **Distributed Collector** → after VAE Decode
4. Enable workers in the UI
5. Run the workflow!

### Distributed Upscaling

![Clipboard Image (2)](https://github.com/user-attachments/assets/ec2548d0-1fc7-4705-801f-3270d720cfce)

1. Load your image
2. Upscale with ESRGAN or similar
3. Connect to **Ultimate SD Upscale Distributed**
4. Configure tile settings
5. Enable workers for faster processing

---

## UI Panel Settings

The Distributed GPU panel includes several configuration options in the Settings section:

### Debug Mode
- **Description**: Enable detailed logging for troubleshooting and monitoring distributed operations
- **Default**: Disabled
- **Usage**: When enabled, detailed debug information is logged to the browser console, including worker status updates, job distribution details, and network communication logs

### Auto-launch Workers on Startup
- **Description**: Automatically launch enabled local workers when ComfyUI starts
- **Default**: Disabled  
- **Usage**: When enabled, any local workers that are marked as "enabled" will be automatically launched in the background when the ComfyUI server starts, eliminating the need to launch each worker manually

### Stop Workers on Master Exit
- **Description**: Automatically stop all managed local workers when the master ComfyUI instance shuts down
- **Default**: Enabled
- **Usage**: When enabled, ensures clean shutdown by stopping all UI-managed worker processes when the main ComfyUI server exits, preventing orphaned background processes

---

## Development

This project is under active development. Contributions are welcome!
