import modal
import os

app = modal.App("upload")
models_volume = modal.Volume.from_name("models") #, create_if_missing=True)

# This works for small files; for larger files upload to HF and download to Modal from HF
local_file = "./iclr26_pretrain_mip_to_mipinfo.log"
remote_path = f"./{local_file}"

@app.local_entrypoint()
def main():
    if not os.path.exists(local_file):
        raise SystemExit(f"File not found {local_file}")

    # Perform the batch upload locally so the local filesystem is used
    with models_volume.batch_upload() as batch:
        batch.put_file(local_file, remote_path)

    print(f"Uploaded {local_file} to volume at {remote_path}")

# import io
# import os
#
# import modal
#
# app = modal.App()
#
#
# @app.function(
#     image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers", "ftfy"),
#     secrets=[modal.Secret.from_name("hf_token")],
#     gpu="any",
# )
# def run_stable_diffusion(prompt: str):
#     from diffusers import StableDiffusionPipeline
#
#     pipe = StableDiffusionPipeline.from_pretrained(
#         "runwayml/stable-diffusion-v1-5",
#         use_auth_token=os.environ["HF_TOKEN"],
#     ).to("cuda")
#
#     image = pipe(prompt, num_inference_steps=10).images[0]
#
#     buf = io.BytesIO()
#     image.save(buf, format="PNG")
#     img_bytes = buf.getvalue()
#
#     return img_bytes
#
#
# @app.local_entrypoint()
# def main():
#     img_bytes = run_stable_diffusion.remote("Wu-Tang Clan climbing Mount Everest")
#     with open("/tmp/output.png", "wb") as f:
#         f.write(img_bytes)
