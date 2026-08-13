import os
import modal
from huggingface_hub import hf_hub_url
import requests

filename = "pretrain_clusters_mip_to_mipinfo.pkl"
app = modal.App("download_hf_into_volume")
models_volume = modal.Volume.from_name("models")
image = modal.Image.debian_slim().pip_install("huggingface-hub", "requests")

# use an absolute path inside the container
MOUNT_PATH = "/root/models"

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("hf_token")],
    volumes={MOUNT_PATH: models_volume},
    timeout=600,
)
def download(repo: str = "skadio/forge", filename: str = filename):
    target_dir = MOUNT_PATH
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    token = os.environ.get("HF_TOKEN")
    url = hf_hub_url(repo_id=repo, filename=filename, repo_type="model")

    headers = {"Authorization": f"Bearer {token}"} if token else None
    tmp_path = target_path + ".part"
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    # atomic replace
    os.replace(tmp_path, target_path)

    print("Downloaded and persisted to volume at", target_path)

@app.local_entrypoint()
def main():
    download.remote()
