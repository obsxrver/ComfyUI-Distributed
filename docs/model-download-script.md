## Automating ComfyUI Model Downloads
> This guide will walk you through creating a shell script to automatically download the necessary models for your ComfyUI workflow, leveraging an advanced Large Language Model (LLM).

1. In ComfyUI (on your local machine), export your workflow as an API workflow
2. Copy the below prompt and upload the API workflow to an LLM **that has access to the internet**

<details>
<summary><strong>📋 Click to expand the full prompt</strong></summary>

```
Create a sh script that will download the models from this workflow into the correct folders. For reference, these are the paths:
base_path: /workspace/ComfyUI
checkpoints: models/checkpoints/
clip: models/clip/
clip_vision: models/clip_vision/
controlnet: models/controlnet/
diffusion_models: models/diffusion_models/
embeddings: models/embeddings/
florence2: models/florence2/
ipadapter: models/ipadapter/
loras: models/loras/
style_models: models/style_models/
text_encoders: models/text_encoders/
unet: models/unet/
upscale_models: models/upscale_models/
vae: models/vae/
---
Important:
Make sure you find the correct URLs for the models online.
Use comfy cli to download the models: `comfy model download --url <URL> [--relative-path <PATH>] [--set-civitai-api-token <TOKEN>] [--set-hf-api-token <TOKEN>]`
Make sure you add `--set-civitai-api-token $CIVITAI_API_TOKEN` for CivitAI download and `--set-hf-api-token $HF_API_TOKEN` for Hugging Face downloads.
---
Example:
#!/bin/bash
# Download from CivitAI
comfy model download --url https://civitai.com/api/download/models/1759168 --relative-path /workspace/ComfyUI/models/checkpoints --set-civitai-api-token $CIVITAI_API_TOKEN
# Download model from Hugging Face
comfy model download --url https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors --relative-path /workspace/ComfyUI/models/unet --set-hf-api-token $HF_API_TOKEN
# If a model in the workflow was in a subfolder
comfy model download --url https://civitai.com/api/download/models/1759168 --relative-path /workspace/ComfyUI/models/checkpoints/SDXL --set-civitai-api-token $CIVITAI_API_TOKEN
```

</details>

3. Review the LLMs output to make sure all download links are correct and save it as a .sh file, for example `download_models.sh`
4. Launch the [ComfyUI Distributed Pod](https://console.runpod.io/deploy?template=m21ynvo8yo&ref=ak218p52) with these Environment Variables:
   - `CIVITAI_API_TOKEN`: [get your token here](https://civitai.com/user/account)
   - `HF_API_TOKEN`: [get your token here](https://huggingface.co/settings/tokens)
5. Upload the .sh file to your Runpod instance, into `/workspace`
6. Then run these commands:
   - `chmod 755 /workspace/download_models.sh`
   - `/workspace/download_models.sh`
7. Confirm each model name (sometimes you might need to rename them to match the name on your local machine)
