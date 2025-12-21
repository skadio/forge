import modal

# Manually create `instances` volume and upload from local via cli
# Manually create mip_to_mipinfo.pkl files and upload to HF via models/local_to_hf.py
# Manually crate `models` volume and download from HF via models/hf_to_modal.py
# Create: modal volume create instances
# Create: modal volume create models
# List: modal volume list
# List: modal volume ls instances
# Upload: modal volume put instances . (inside /data/instances)
# Rename: modal volume rename data instances
# Remove: modal volume rm instances configs -r
instances_volume = modal.Volume.from_name("instances")
models_volume = modal.Volume.from_name("models")

# Create a modal image and install libraries
# Copy the current folder/repo, except main.py file
forge_image = modal.Image.debian_slim(python_version="3.12").run_commands(
    "apt-get update",
    "pip install gurobipy numpy huggingface-hub pandas pyyaml scikit-learn scipy",
    "pip install torch torch-geometric tqdm vector-quantize-pytorch"
).add_local_dir(".", "/root/", ignore=["./main.py",  # This will be copied when running modal on main.py
                                       "./data/instances/", "./models",  # these will come from Volumes
                                       "./experiments", "./tests",  # not needed
                                       "./.gitignore", "./CHANGELOG.txt", "./LICENSE", "./MANIFEST.in",
                                       "./pyproject.toml", "./README.md", "./.git", "./.idea", "./__pycache__",
                                       "./data/__pycache__" "./forge/__pycache__", "./experiments/__pycache__"])

# Create Modal app
app = modal.App("Forge-ICLR-Pretrain", image=forge_image)

# Nvidia B200 # $6.25 / h
# Nvidia H200 # $4.54 / h
# Nvidia H100 # $3.95 / h
# Nvidia A100, 80 GB # $2.50 / h
# Nvidia A100, 40 GB # $2.10 / h
# Nvidia L40S # $1.95 / h
# Nvidia A10 # $1.10 / h
# Nvidia L4 # $0.80 / h
# Nvidia T4 # $0.59 / h
@app.function(volumes={"/root/data/instances": instances_volume,
                       "/root/models/": models_volume},
              timeout=86000,
              gpu="H200")
def run():
    import os, subprocess
    current_dir = os.getcwd()
    # parent_dir = os.path.dirname(current_dir)
    print("Current directory:", current_dir)
    subprocess.run(["ls", "-al", current_dir])

    from forge.embeddings import Forge
    from forge.pipeline import pretrain

    forge = Forge(train_config_yaml="./forge/configs/train_config.yaml")

    config="iclr26_pretrain"
    pretrain(forge=forge,
             input_mip_folder="/root/data/instances/",
             input_mip_instances_file=f"/root/data/configs/{config}.txt",
             output_mip_to_mipinfo_pkl=f"/root/models/{config}_mip_to_mipinfo.pkl",
             input_mip_to_mipinfo_pkl=f"/root/models/{config}_mip_to_mipinfo.pkl",
             output_forge_pretrained_pkl=f"/root/models/{config}_pretrained.pkl",
             output_log_file=f"/root/models/{config}_pretrained.log")

# > modal run main.py
# --detach flag to run in background, continue even terminal is closed
@app.local_entrypoint()
def main():
    # print(run.local())
    run.remote()

# Define CUDA base image tag
# cuda_version = "12.4.1"
# flavor = "devel"
# operating_sys = "ubuntu22.04"
# tag = f"{cuda_version}-{flavor}-{operating_sys}"

# # Build Modal image with Miniconda + DGL
# image = (
#     # modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
#     modal.Image.debian_slim(python_version="3.11")
#     .apt_install("wget")  # needed to fetch Miniconda
#     .run_commands(
#         # Install Miniconda
#         "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh",
#         "bash miniconda.sh -b -p /opt/conda",
#         "rm miniconda.sh",
#
#         # Accept Anaconda ToS for required channels
#         # "/opt/conda/bin/conda config --add channels defaults",
#         "/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main",
#         "/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r",
#
#         # Update conda
#         "/opt/conda/bin/conda update -n base -c defaults -y conda",
#
#         # Force base env to Python 3.11 (otherwise 3.13 will clash with DGL requirement)
#         "/opt/conda/bin/conda install -n base -y python=3.11",
#
#         # Create conda environment with python version
#         # "/opt/conda/bin/conda create -y -n myforge_311 python=3.11",
#
#         # Install DGL inside that env
#         # "/opt/conda/bin/conda run -n myforge_311 conda install -y -c dglteam/label/th24_cu124 dgl",
#         "/opt/conda/bin/conda install -y -c dglteam/label/th24_cu124 dgl",
#
#         # Install PyTorch within that env (rather than doing conda activate myforge_311)
#         # "/opt/conda/bin/conda run -n myforge_311 conda install -y pytorch==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia",
#         "/opt/conda/bin/conda install -y pytorch==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia",
#
#         # Install other libs
#         # "/opt/conda/bin/conda run -n myforge_311 pip install numpy pandas scikit-learn scipy pyyaml pydantic gurobipy category-encoders einops googledrivedownloader ogb"
#         "/opt/conda/bin/pip install numpy pandas scikit-learn scipy pyyaml pydantic gurobipy category-encoders einops googledrivedownloader ogb"
#     )
#     .env({"PATH": "/opt/conda/bin:" + "$PATH"})
# )
