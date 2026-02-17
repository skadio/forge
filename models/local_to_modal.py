import modal
import os

app = modal.App("upload")
models_volume = modal.Volume.from_name("models") #, create_if_missing=True)

# This only works for SMALL files
# For large files (GBs), first upload to HF, and then download from HF to Modal
local_file = "iclr_pretrain_clusters_mip_to_mipinfo.log"
remote_path = f"./{local_file}"

@app.local_entrypoint()
def main():
    if not os.path.exists(local_file):
        raise SystemExit(f"File not found {local_file}")

    # Perform the batch upload locally so the local filesystem is used
    with models_volume.batch_upload() as batch:
        batch.put_file(local_file, remote_path)

    print(f"Uploaded {local_file} to volume at {remote_path}")
