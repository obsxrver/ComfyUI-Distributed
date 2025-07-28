## Local workers
> Local Workers: These are added automatically on first launch, but you can add them manually if you need.

1. **Open** the Distributed GPU panel.
2. **Click** "Add Worker" in the UI.
3. **Configure** your local worker:
   - **Name**: A descriptive name for the worker (e.g., "Studio PC 1")
   - **Port**: A unique port number for this worker (e.g., 8189, 8190...).
   - **CUDA Device**: The GPU index from `nvidia-smi` (e.g., 0, 1).
   - **Extra Args**: Optional ComfyUI arguments for this specific worker.
4. **Save** and  launch the local worker.

## Remote workers
> Remote Workers: ComfyUI instances running on completely different computers on your network. These allow you to harness GPU power from other machines. Remote workers must be manually started on their respective computers and are connected via IP address.

**On the Remote Worker Machine:**
1. **Launch** the master (main ComfyUI instance) with the `--listen --enable-cors-header` arguments. ⚠️ **Required!**
2. **Add** workers in the UI panel if the remote machine has more than one GPU.
   - Make sure that they also have `--listen` set in `Extra Args`.
   - Then launch them.
3. **Open** the configured worker port(s) (e.g., 8189, 8190) in the remote worker's firewall.
  
**On the Main Machine:**
1. **Open** the Distributed GPU panel (sidebar on the left).
2. **Click** "Add Worker."
3. **Choose** "Remote".
4. **Configure** your remote worker:
   - **Name**: A descriptive name for the worker (e.g., "Server Rack GPU 0")
   - **Host**: The remote worker's IP address.
   - **Port**: The port number used when launching ComfyUI on the remote worker (e.g., 8189).
5. **Save** the remote worker configuration.
  
## Cloud workers
> Cloud Workers: ComfyUI instances running on a cloud service like [Runpod](https://get.runpod.io/0bw29uf3ug0p). 

### Deploy Cloud Worker on Runpod

**On Runpod:**
1. Register a [Runpod](https://get.runpod.io/0bw29uf3ug0p) account.
2. On Runpod, go to Storage > New Network Volume and create a volume which will store the models you need.
3. Now go to Pods and find a suitable GPU for your workflows. 
> To use the ComfyUI-Distributed-Pod, you will need to filter instances by CUDA 12.8.
4. Choose the `ComfyUI-Distributed-Pod` template and make sure your network drive is mounted.
5. Launch your pod.
6. Access your pod using JupyterLab.
7. Download models into /workspaces/ComfyUI/models/ (these will remain on your network drive even after you terminate the pod).
> You can use [this guide](model-download-script.md) to make this process easy for you. It will generate a shell script that will automatically download the models you need for a given workflow.
8. If using your own template, make sure you launch ComfyUI with the `--enable-cors-header` argument and you git clone ComfyUI-Distributed into custom_nodes. ⚠️ **Required!**
9. Download any additional custom nodes using ComfyUI manager

**On the Main Machine:**
1. Launch a Cloudflare tunnel
   - Download from here `https://github.com/cloudflare/cloudflared/releases`
	- Then run for example: `cloudflared-windows-amd64.exe tunnel --url http://localhost:8188`
2. Copy the Cloudflare address
3. **Open** the Distributed GPU panel (sidebar on the left).
4. **Edit** the Master's host address and replace it with the Cloudflare address.
   - **Click** "Add Worker."
   - **Choose** "Cloud".
   - **Configure** your cloud worker:
     - **Host**: The Runpod address
     - **Port**: 443
   - **Save** the remote worker configuration.

### Deploy Cloud Worker on Other Platforms

**On the Cloud Worker machine:**
   - Your cloud worker container needs to have the same models and custom nodes as the workflow you want to run on your local machine.
   - If your cloud platform doesn't provide a secure connection, use Cloudflare to create a tunnel for the worker. Each GPU needs their own tunnel for their respective port.
	   - For example: `./cloudflared tunnel --url http://localhost:8188`
1. **Launch** ComfyUI with the `--listen --enable-cors-header` arguments. ⚠️ **Required!**
2. **Add** workers in the UI panel if the remote machine has more than one GPU.
   - Make sure that they also have `--listen` set in `Extra Args`.
   - Then launch them.
  
**On the Main Machine:**
1. **Launch** a Cloudflare tunnel on your local machine.
   - Download from here `https://github.com/cloudflare/cloudflared/releases`
   - For example: `cloudflared-windows-amd64.exe tunnel --url http://localhost:8188`
2. **Copy** the Cloudflare address
3. **Open** the Distributed GPU panel (sidebar on the left).
4. **Edit** the Master's host address and replace it with the Cloudflare address.
5. **Click** "Add Worker."
6. **Choose** "Cloud".
7. **Configure** your cloud worker:
   - **Host**: The remote worker's IP address/domain
   - **Port**: 443
8. **Save** the remote worker configuration.