**Accelerated Video Upscaler On Runpod:**

1. Use the [ComfyUI Distributed Pod](https://console.runpod.io/deploy?template=m21ynvo8yo&ref=0bw29uf3ug0p) template.
2. Filter instances by CUDA 12.8 (add filter in Additional Filters at the top of the page).
3. Choose 4x 5090s
4. Press Edit Template to configure the pod's Environment Variables:
	- CIVITAI_API_TOKEN: [get your token here](https://civitai.com/user/account)
	- HF_API_TOKEN: [get your token here](https://huggingface.co/settings/tokens)
	- SAGE_ATTENTION: optional optimisation (set to true/false). Recommended for this workflow.
	- PRESET_VIDEO_UPSCALER: set to true. This will download everything you need.
5. Deploy your pod.
6. Once pod setup is complete, connect to ComfyUI running on your pod.
7. Open the GPU panel on the left.
> If you set SAGE_ATTENTION to true, add "--use-sage-attention" to Extra Args on the workers.
8. Launch the workers.
9. Upload video, add prompt and run workflow.
10. Right-click the Video Combine node and click Save Preview to save the video.
