## Automating ComfyUI Model Downloads

This guide will walk you through creating a shell script to automatically download the necessary models for your ComfyUI workflow, leveraging an advanced Large Language Model (LLM).

1. Launched the Pod with these Environment Variables:
  - CIVITAI_API_TOKEN ([get your token here](https://civitai.com/user/account))
  - HF_API_TOKEN ([get your token here](https://huggingface.co/settings/tokens))
3. In ComfyUI, export your workflow as an API workflow
4. Copy the below prompt and upload the API workflow to a LLM **that has access to the internet**
```
Create a sh script that will download the models from this workflow into the correct folders. For reference, these are the paths:
comfyui:
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
Make sure you find the correct URLs for the models online.
Use comfy cli to download the models: comfy model download --url <URL> ?[--relative-path <PATH>]
```
5. Review the LLMs output to make sure all download links are correct and save it as a .sh file
6. Upload it onto your Runpod instance
7. Run the file. For example: `./download_comfyui_models.sh`
