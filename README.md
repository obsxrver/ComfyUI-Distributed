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

2. **Restart ComfyUI** - the extension will create a default configuration file on first load

---

## Quick Start

### Adding Local Workers
![Clipboard Image (4)](https://github.com/user-attachments/assets/9c1d6d0e-3fd1-43e3-97c4-7c6bf2952b19)

1. **Open** the Distributed GPU panel (sidebar on the left)
2. **Click** "Add Worker" in the UI
3. **Configure** your worker:
   - **Name**: Descriptive name for the worker
   - **Host**: `localhost` for same machine
   - **Port**: Unique port number (e.g., 8189, 8190...)
   - **CUDA Device**: GPU index from `nvidia-smi`
   - **Extra Args**: Optional ComfyUI arguments
4. **Save** and optionally launch the worker

### Adding Remote Workers

1. **Follow** the same steps as above, but:
2. **Add** `--listen` to the 'Extra Args' field ⚠️ **Required!**
3. **Open** the worker port in your firewall
4. **Launch** ComfyUI on the remote machine
5. **Add** the worker using the remote machine's IP address

> **Tip**: If the remote machine has multiple GPUs, create multiple workers with different CUDA devices and ports

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
3. Configure tile settings based on your VRAM
4. Enable workers for faster processing

---

## UI Features

### Distributed GPU Panel

The control center for your distributed setup:

| Feature | Description |
|---------|-------------|
| **Worker Status** | Real-time status indicators |
| **Launch/Stop** | Control individual workers |
| **Clear Memory** | Free VRAM on all workers |
| **Interrupt** | Stop current processing |
| **Auto-detect IP** | Find best network interface |
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
