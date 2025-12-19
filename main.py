import modal

# Manually create a mount of all instances from command line
# Create: modal volume create instances
# List: modal volume list
# Upload: modal volume put instances . (inside forge/data/instances)
# LS: modal volume ls data
# modal volume rename data instances
#  modal volume rm instances configs -r
instances_volume = modal.Volume.from_name("instances")

# Create a modal image and install libraries
# Copy the current folder/repo, except main.py fiel
forge_image = modal.Image.debian_slim(python_version="3.12").run_commands(
    "apt-get update",
    "pip install gurobipy numpy huggingface-hub pandas pyyaml scikit-learn scipy",
    "pip install torch torch-geometric tqdm vector-quantize-pytorch"
).add_local_dir(".", "/root/", ignore=["./data/instances/", "./tests", "./main.py",
                                       "./.git", "./.idea", "./__pycache__", "./data/__pycache__"
                                       "./forge/__pycache__", "./experiments/__pycache__"])

# Create Modal app
app = modal.App("Forge", image=forge_image)


@app.function(volumes={"/root/data/instances": instances_volume})
def run():
    import os, subprocess

    current_dir = os.getcwd()
    # parent_dir = os.path.dirname(current_dir)
    print("Current directory:", current_dir)
    subprocess.run(["ls", "-l", current_dir])

    from forge.embeddings import Forge
    from forge.pipeline import pretrain

    forge = Forge(train_config_yaml="./forge/configs/train_config.yaml")


# > modal run main.py
@app.local_entrypoint()
def main():
    # run locally
    # print(run.local())

    # # run remotely on Modal
    # print(run.remote())
    list_data.remote()

    # # run remotely on Modal in parallel
    # total = 0
    # for ret in f.map(range(10)):
    #     total += ret
    # print(total)

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


# modal volume create data
# Created Volume 'data' in environment 'None'.
# Code example:
# @app.function(volumes={"/my_vol": modal.Volume.from_name("data")})
# def some_func():
#     os.listdir("/my_vol")
# modal volume list
# modal volume put data . (inside forge/data/)
# modal volume ls data
