import argparse
import os
import shutil
import zipfile
from huggingface_hub import hf_hub_download


def download_file(repo_id: str, filename: str, output_path: str, repo_type: str = "dataset", token: str | None = None):
    print(f"<< Downloading {filename} -> {output_path}")
    cached_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, token=token)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    shutil.copy(cached_path, output_path)
    print(f">> DONE! Downloading {filename} -> {output_path}")


def unzip_file(zip_path: str, unzip_folder: str):
    print(f"<< Extracting {zip_path} -> {unzip_folder}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_folder)
    print(f"<< DONE! Extracting {zip_path} -> {unzip_folder}")


if __name__ == "__main__":
    # url = "https://huggingface.co/datasets/skadio/forge"
    parser = argparse.ArgumentParser(description="Download a file from Hugging Face Forge Dataset")
    parser.add_argument("--repo", default="skadio/forge", help="Hugging Face repo id")
    parser.add_argument("--zip_filename", default="instances.zip", help="Zip file in the repo to download")
    parser.add_argument("--unzip_folder", default="./instances", help="Folder to unzip output")
    parser.add_argument("--token", default=None, help="HF token if repo is private")
    args = parser.parse_args()

    download_file(repo_id=args.repo, filename=args.zip_filename, output_path=args.zip_filename, token=args.token)
    unzip_file(zip_path=args.zip_filename, unzip_folder=args.unzip_folder)
