## Prompt for LLM - Automating ComfyUI Model Downloads

This guide will walk you through creating a shell script to automatically download the necessary models for your ComfyUI workflow, leveraging an advanced Large Language Model (LLM).

1. Export your workflow as an API workflow
2. Copy the below prompt and upload the API workflow to a LLM that has access to the internet (like Gemini Flash)
3. Review (make sure all downloads are correct) and save the output from the LLM as a .sh file
4. Upload it onto your Runpod instance
5. Run the file. For example: `./download_comfyui_models.sh`

```
Create a sh script that will download the models from this workflow into the correct folders. For reference, these are the paths:
comfyui:
base_path: /workspace/ComfyUI
BiRefNet: models/BiRefNet/
checkpoints: models/checkpoints/
clip: models/clip/
clip_vision: models/clip_vision/
configs: models/configs/
controlnet: models/controlnet/
diffusers: models/diffusers/
diffusion_models: models/diffusion_models/
embeddings: models/embeddings/
florence2: models/florence2/
facerestore_models: models/facerestore_models/
gligen: models/gligen/
grounding-dino: models/grounding-dino/
hypernetworks: models/hypernetworks/
ipadapter: models/ipadapter/
lama: models/lama/
loras: models/loras/
onnx: models/onnx/
photomaker: models/photomaker/
RMBG: models/RMBG/
sams: models/sams/
style_models: models/style_models/
text_encoders: models/text_encoders/
unet: models/unet/
upscale_models: models/upscale_models/
vae: models/vae/
vae_approx: models/vae_approx/
vitmatte: models/vitmatte/
```