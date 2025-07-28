<div align="center">
<img width="320" src="https://github.com/user-attachments/assets/537d38cc-2e34-4527-bca7-0d79f4120cce" />
<br><br>
<a href="">📺Video Tutorial</a> |
<a href="/docs/worker-setup-guides.md">📝Setup Guides</a> | 
<a href="/workflows">⚙️Workflows</a> |
<a href="https://buymeacoffee.com/robertvoy">🎁Donation</a> 
<br><br>
</div>

> **A powerful extension for ComfyUI that enables distributed and parallel processing across multiple GPUs and machines. Generate more images and videos and accelerate your upscaling workflows by leveraging all available GPU resources in your network.**

![Clipboard Image (3)](https://github.com/user-attachments/assets/19fdd1be-8f6e-4df5-bcd9-538ef566fa82)

---

## Key Features

#### Parallel Workflow Processing
- Run workflows on multiple GPUs simultaneously with varied seeds
- Scale output with more workers
- Supports images and videos

#### Distributed Upscaling
- Accelerate Ultimate SD Upscale by distributing tiles across GPUs
- Intelligent distribution
- Handles single images and batches

#### Ease of Use
- Auto-setup local workers; easily add remote/cloud ones
- Convert any workflow to distributed with 2 nodes
- JSON configuration with UI controls

---

## Requirements

- ComfyUI
> Note: Desktop app not supported currently
- Multiple GPUs
- That's it

---

## Installation

1. **Clone this repository** into your ComfyUI custom nodes directory:
   ```bash
   git clone https://github.com/robertvoy/ComfyUI-Distributed.git
   ```

2. **Restart ComfyUI** - If you'll be using remote/cloud workers, add `--enable-cors-header` to your launch arguments on the master

3. Read the [setup guide](https://github.com/robertvoy/ComfyUI-Distributed/blob/feature/cloud-workers/docs/worker-setup-guides.md) for adding workers

---

## Workflow Examples

### Basic Parallel Generation
> [Download workflow](/workflows/distributed-txt2img.json)
![Clipboard Image (1)](https://github.com/user-attachments/assets/e8e46d97-d698-4c18-b4e5-1e1a2f4f7da3)

1. Create your normal ComfyUI workflow
2. Add **Distributed Seed** → connect to sampler's seed
3. Add **Distributed Collector** → after VAE Decode
4. Enable workers in the UI
5. Run the workflow!

### Parallel WAN Generation

> [Download workflow](/workflows/distributed-wan.json)
> 
### Distributed Upscaling

![Clipboard Image (2)](https://github.com/user-attachments/assets/ec2548d0-1fc7-4705-801f-3270d720cfce)

> [Download workflow](/workflows/distributed-txt2img.json)

1. Load your image
2. Upscale with ESRGAN or similar
3. Connect to **Ultimate SD Upscale Distributed**
4. Configure tile settings
5. Enable workers for faster processing

---

## FAQ

<details>
<summary>Does it combine VRAM of multiple GPUs?</summary>
No, it does not combine VRAM of multiple GPUs.
</details>

<details>
<summary>Does it speed up the generation of a single image or video?</summary>
No, it does not speed up the generation of a single image or video. Instead, it enables the generation of more images or videos simultaneously. However, it can speed up the upscaling of a single image when using the Ultimate SD Upscale Distributed feature.
</details>

<details>
<summary>Does it work with the ComfyUI desktop app?</summary>
Currently, it is not compatible with the ComfyUI desktop app.
</details>

<details>
<summary>Can I combine my RTX 5090 with a GTX 980 to get faster results?</summary>
Yes, you can combine different GPUs, but performance is optimized when using similar GPUs. A significant performance imbalance between GPUs may cause bottlenecks. For upscaling, setting `static_distribution` to `false` allows the faster GPU to handle more processing, which can mitigate some bottlenecks. Note that this setting only applies to upscaling tasks.
</details>

<details>
<summary>Does this work with cloud providers?</summary>
Yes, it is compatible with cloud providers. Refer to the setup guides for detailed instructions.
</details>

<details>
<summary>Can I make this work with my Docker setup?</summary>
Yes, it is compatible with Docker setups, but you will need to configure your Docker environment yourself. Unfortunately, assistance with Docker configuration is not provided.
</details>

---

## Disclaimer

By using this software, you agree to be bound by the terms of this Disclaimer. The software is provided “as is” without any warranties, express or implied, including but not limited to merchantability, fitness for a particular purpose, or non-infringement. The developers and copyright holders are not liable for any claims, damages, or liabilities—whether in contract, tort, or otherwise—arising from the use, modification, distribution, or other dealings with the software.

Users are solely responsible for ensuring their use of the software, including any content it generates, complies with all applicable laws and regulations in their jurisdiction. The developers are not responsible for any legal violations resulting from user actions.

Additionally, users are responsible for the security and integrity of their networks, including protecting against unauthorized access, hacking, data breaches, or loss. The developers assume no liability for any damages, losses, or incidents arising from improper configuration, misuse, or external threats to user environments.

---

## Support the Project
If my custom nodes have added value to your workflow, consider fueling future development with a coffee! Your support helps keep this project thriving. Buy me a coffee at: https://buymeacoffee.com/robertvoy
