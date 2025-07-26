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

#### Convenience
- Automatically set up your local workers and easily add remote and cloud workers
- Manage the workers through the UI panel
- Easily convert any workflow to a distributed one, using no more than 2 nodes

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

## FAQ

Does it combine VRAM of multiple GPUs?
No.

Does it speed up the generation of a single image/video?
No, it gives you more images/videos, rather than a faster single one. However, it does speed up the upscaling of a single image using the Ultimate SD Upscale Distributed.

Does it work with ComfyUI desktop app?
Not currently.

Can I combine my RTX 5090 with a GTX 980 to get faster results?
You can, but this works best with cards that are similar. If you have a large imbalance between GPUs, you will run into bottlenecks. Although you could still benefit from Ultimate SD Upscale Distributed with static_distribution set to false. This will allow the faster card to process more. Note this only works for upscaling.

Does this work with cloud providers?
Yes, see setup guides.

Can I make this work with my Docker setup?
Yes, it does work, but you need to know how to set up your Docker environment. I won't be able to assist you with that.

---

## Disclaimer

This software is provided “as is” without any warranties, express or implied, including but not limited to merchantability, fitness for a particular purpose, or non-infringement. The developers and copyright holders are not liable for any claims, damages, or liabilities—whether in contract, tort, or otherwise—arising from the use, modification, distribution, or other dealings with the software.

Users are solely responsible for ensuring their use of the software, including any content it generates, complies with all applicable laws and regulations in their jurisdiction. The developers are not responsible for any legal violations resulting from user actions.

Additionally, users are responsible for the security and integrity of their networks, including protecting against unauthorized access, hacking, data breaches, or loss. The developers assume no liability for any damages, losses, or incidents arising from improper configuration, misuse, or external threats to user environments.

---

## Support the Project
If my custom nodes have added value to your workflow, consider fueling future development with a coffee! Your support helps keep this project thriving. Buy me a coffee at: https://buymeacoffee.com/robertvoy
