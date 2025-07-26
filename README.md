<div align="center">
<img width="460" alt="distributed-logo" src="https://github.com/user-attachments/assets/e4604f86-f663-4672-b568-6f3df4282a1e" />
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
- Run workflows on multiple GPUs simultaneously with varied seeds
- Generate multiple outputs in the time of one
- Scale output with more workers
- Supports images and videos

#### Distributed Upscaling
- Accelerate Ultimate SD Upscale by distributing tiles across GPUs
- Intelligent distribution for asymmetric setups
- Static distribution for similar GPUs
- Handles single images and batches

#### Convenience
- Auto-setup local workers; easily add remote/cloud ones
- Manage workers via UI panel
- Convert workflows to distributed with less than 2 nodes
- JSON configuration with UI controls

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

## Nodes

### Distributed Collector
> Collects and combines results from distributed processing

**Usage**: Place after any image generation node to enable distributed processing. Works automatically when workers are enabled.

### Distributed Seed
> Ensures unique seeds across distributed workers for varied generations

**Usage**: Connect to any seed input. Each worker automatically receives an offset seed to ensure randomisation. Alternatively, you can connect a seed node directly to this node's seed input, which will automatically handle seed offsetting across all workers.

### Image Batch Divider
> Divides image batches, used video

**Usage**: Place after the Distributed Collector and set the divide_by to the number of GPUs you are using (including the master)
  
### Ultimate SD Upscale Distributed
> Distributed version of Ultimate SD Upscale that processes tiles across multiple GPUs, making upscaling much faster

**Usage**:
1. Upscale your image with a regular upscale model (ESRGAN, etc.)
2. Feed the upscaled image into this node
3. Configure tile settings
4. Enable workers for fast processing

---

## Disclaimer

By using this software, you agree to be bound by the terms of this Disclaimer. The software is provided “as is” without any warranties, express or implied, including but not limited to merchantability, fitness for a particular purpose, or non-infringement. The developers and copyright holders are not liable for any claims, damages, or liabilities—whether in contract, tort, or otherwise—arising from the use, modification, distribution, or other dealings with the software.

Users are solely responsible for ensuring their use of the software, including any content it generates, complies with all applicable laws and regulations in their jurisdiction. The developers are not responsible for any legal violations resulting from user actions.

Additionally, users are responsible for the security and integrity of their networks, including protecting against unauthorized access, hacking, data breaches, or loss. The developers assume no liability for any damages, losses, or incidents arising from improper configuration, misuse, or external threats to user environments.

---

## Support the Project
If my custom nodes have added value to your workflow, consider fueling future development with a coffee! Your support helps keep this project thriving. Buy me a coffee at: https://buymeacoffee.com/robertvoy
