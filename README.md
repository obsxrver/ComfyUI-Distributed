# ComfyUI-Distributed

![Clipboard Image (3)](https://github.com/user-attachments/assets/19fdd1be-8f6e-4df5-bcd9-538ef566fa82)

> **Supercharge your ComfyUI workflows with distributed GPU processing**

A powerful extension for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that enables distributed processing across multiple GPUs and machines. Speed up your image generation and upscaling workflows by leveraging all available GPU resources in your network.

---

## Features

- **Distributed Processing** - Split workloads across multiple GPUs automatically
- **Distributed Upscaling** - Tile-based upscaling with intelligent work distribution
- **Automatic Worker Management** - Launch and monitor workers from the UI
- **Network Support** - Use GPUs across different machines on your network
- **Real-time Monitoring** - Track worker status and performance from the UI
- **Easy Configuration** - JSON-based configuration with UI controls
- **Memory Management** - Built-in VRAM clearing and optimization

---

## Requirements

- ComfyUI installation
- PyTorch with CUDA support
- Python 3.8+

---

## Installation

1. **Clone this repository** into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-Distributed.git
   ```

2. **Restart ComfyUI** - add `--enable-cors-header` to your launch argumenets

---

## Quick Start

### Adding Local Workers
![Distributed GPU Panel](https://github.com/user-attachments/assets/9c1d6d0e-3fd1-43e3-97c4-7c6bf2952b19)
*The Distributed GPU panel can be found in the sidebar on the left.*

1. **Open** the Distributed GPU panel.
2. **Click** "Add Worker" in the UI.
3. **Configure** your local worker:
   - **Name**: A descriptive name for the worker (e.g., "My Gaming PC GPU 0")
   - **Port**: A unique port number for this worker (e.g., 8189, 8190...).
   - **CUDA Device**: The GPU index from `nvidia-smi` (e.g., 0, 1).
   - **Extra Args**: Optional ComfyUI arguments for this specific worker.
4. **Save** and optionally launch the local worker.

### Adding Remote Workers

1. **On the Remote Worker Machine:**
   - **Launch** ComfyUI with the `--listen --enable-cors-header` arguments. ⚠️ **Required!**
   - **Add** workers in the UI panel if the remote machine has more than one GPU.
      - Make sure that they also have `--listen` set in `Extra Args`.
      - Then launch them.
   - **Open** the configured worker port(s) (e.g., 8189, 8190) in the remote worker's firewall.
  
2. **On the Main Machine:**
   - **Open** the Distributed GPU panel (sidebar on the left).
   - **Click** "Add Worker."
   - **Enable** "Remote Worker" checkbox.
   - **Configure** your remote worker:
     - **Name**: A descriptive name for the worker (e.g., "Server Rack GPU 0")
     - **Host**: The remote worker's IP address.
     - **Port**: The port number used when launching ComfyUI on the remote worker (e.g., 8189).
   - **Save** the remote worker configuration.

### Configuration Tips

| Setting | Description | Example |
|---------|-------------|---------|
| **CUDA Devices** | Use `nvidia-smi` to see GPU indices | 0, 1, 2... |
| **Ports** | Each worker needs a unique port | 8189, 8190... |
| **Extra Args** | Additional ComfyUI arguments | See below |

**Common Extra Args:**
- `--lowvram` - For GPUs with less memory
- `--listen` - **Required** for remote workers
- `--highvram` - For GPUs with 12GB+ VRAM

---

## Nodes

### Distributed Collector
> Collects and combines results from distributed processing

**Usage**: Place after any image generation node to enable distributed processing. Works automatically when workers are enabled.

### Distributed Seed
> Ensures unique seeds across distributed workers for varied generation

**Usage**: Connect to any seed input. Each worker automatically receives an offset seed for variety.

### Ultimate SD Upscale Distributed
> Distributed version of Ultimate SD Upscale that processes tiles across multiple GPUs

**Usage**:
1. Upscale your image with a regular upscale model (ESRGAN, etc.)
2. Feed the upscaled image into this node
3. Configure tile settings
4. Enable workers for faster processing

---

## UI Features

### Distributed GPU Panel

The control center for your distributed setup:

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

## Troubleshooting

### Common Issues

<details>
<summary><b>Workers won't start</b></summary>

- Check ports are not in use: `netstat -an | grep 8189`
- Verify CUDA device exists: `nvidia-smi`
- Check ComfyUI path in worker logs
</details>

<details>
<summary><b>"Worker not managed by UI" message</b></summary>

- Worker was started outside the UI
- Stop the worker manually and use the UI to relaunch
</details>

<details>
<summary><b>Images not combining properly</b></summary>

- Ensure all workers have the same models loaded
- Check that custom nodes are installed on all workers
</details>

<details>
<summary><b>Network connection issues</b></summary>

- Check firewall settings for required ports
- Verify master IP is accessible: `ping 192.168.1.100`
- Ensure same ComfyUI version on all machines
- Ensure ComfyUI-Distributed is installed on remote workers
</details>

### Debug Mode

Enable detailed logging for troubleshooting:

1. **UI Method**: Settings → Debug Mode → Enable
2. **Check** console output for detailed logs

---

## Development

This project is under active development. Contributions are welcome!

### Planned Features

- [ ] Remote worker control via SSH
- [ ] View remote worker logs in UI
- [ ] Improve worker timeout logic
