## Automating ComfyUI Model Downloads

This guide will walk you through creating a shell script to automatically download the necessary models for your ComfyUI workflow, leveraging an advanced Large Language Model (LLM).

1. Make sure you launched the Pod with these Environment Variables: CIVITAI_API_TOKEN ([get your token here](https://civitai.com/user/account)) and HF_API_TOKEN ([get your token here](https://huggingface.co/settings/tokens))
2. Export your workflow as an API workflow
3. Copy the below prompt and upload the API workflow to a LLM that has access to the internet
4. Review (make sure all downloads are correct) and save the output from the LLM as a .sh file
5. Upload it onto your Runpod instance
6. Run the file. For example: `./download_comfyui_models.sh`

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
