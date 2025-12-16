import argparse
import os
import shutil
from huggingface_hub import hf_hub_download


def download_file(repo_id: str, filename: str, output_path: str, repo_type: str = "dataset", token: str | None = None):
    # Download from HF cache (will raise if file not found)
    cached_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, token=token)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    shutil.copy(cached_path, output_path)
    print(f"Saved {filename} -> {output_path}")

if __name__ == "__main__":
    url = "https://huggingface.co/datasets/skadio/forge/blob/main/data.zip"
    parser = argparse.ArgumentParser(description="Download a file from a Hugging Face dataset repo")
    parser.add_argument("--repo", default="skadio/forge", help="Hugging Face repo id (e.g. skadio/forge)")
    parser.add_argument("--filename", default="data.zip", help="Filename in the repo to download")
    parser.add_argument("--output", default="./data.zip", help="Local output path (e.g. ./data.zip)")
    parser.add_argument("--token", default=None, help="HF token if repo is private")
    args = parser.parse_args()

    download_file(repo_id=args.repo, filename=args.filename, output_path=args.output, token=args.token)

