from huggingface_hub import upload_file, HfApi, upload_folder, login
import argparse
import os

from huggingface_hub.errors import RepositoryNotFoundError

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a file to Hugging Face models")
    parser.add_argument("--repo", default="skadio/forge", help="Hugging Face repo id")
    parser.add_argument("--file", required=True, help="Local file path to upload")
    parser.add_argument("--path_in_repo", default=None, help="Target path/name inside the repo")
    parser.add_argument("--token", default=None, help="HF token if repo is private")
    args = parser.parse_args()

    # optional HF login
    # login()

    path_in_repo = args.path_in_repo or os.path.basename(args.file)
    print(f"<< Uploading ` {args.file} ` -> ` {args.repo}/{path_in_repo} `")

    upload_file(path_or_fileobj=args.file, path_in_repo=path_in_repo,
                repo_id=args.repo, repo_type="model", token=args.token)

    print(f"<< Done! Uploading ` {args.file} ` -> ` {args.repo}/{path_in_repo} `")