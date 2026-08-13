import argparse
import os
import shutil
import sys

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from forge.embeddings import Forge
from forge.pipeline import mip_to_embeddings
from forge.utils import Constants

# Hugging Face pretrained model repository
HF_REPO_ID = "skadio/forge"
HF_MODEL_FILENAME = "forge_pretrain_trained.pkl"
HF_MODEL_URL = f"https://huggingface.co/{HF_REPO_ID}"

# Anchored defaults, independent of the current working directory
DEFAULT_TRAIN_CONFIG_YAML = Constants.default_train_config_yaml
DEFAULT_INPUT_FORGE_PKL = os.path.join(Constants.MODELS_DIR, Constants._FORGE_PKL_NAME)
DEFAULT_INPUT_MIPS = Constants.DATA_INSTANCE_DIR
DEFAULT_INPUT_MIP_INSTANCES_FILE = os.path.join(os.path.dirname(Constants.DATA_INSTANCE_DIR), "configs", "all.txt")
DEFAULT_OUTPUT_MIP_TO_EMBEDDINGS_PKL = os.path.join(Constants.MODELS_DIR, Constants._EMBEDDINGS_NAME)


def _loud(*lines):
    """Print a loud, banner-style message."""
    print("\n" + "=" * 70)
    for line in lines:
        print(f"  {line}")
    print("=" * 70 + "\n")


def _hf_token():
    """Return the Hugging Face token from the environment, or None."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _ensure_hf_token():
    """Fail loudly with setup instructions if no Hugging Face token is set."""
    token = _hf_token()
    if token:
        return token
    _loud("HUGGING FACE TOKEN NOT FOUND",
          "The pretrained Forge model is gated, so a Hugging Face token is required.",
          "1. Create a token at https://huggingface.co/settings/tokens",
          "2. Export it, e.g.  export HF_TOKEN=hf_...",
          f"3. First request access to the model at {HF_MODEL_URL}",
          "Then re-run this command.")
    sys.exit(1)


def _check_hf_access(token):
    """Verify the token can access the gated model repo. Exit with guidance if not."""
    try:
        HfApi().get_paths_info(repo_id=HF_REPO_ID, paths=[HF_MODEL_FILENAME], token=token)
    except GatedRepoError:
        _loud("NO ACCESS TO GATED MODEL",
              f"The Hugging Face token is set, but this account cannot access {HF_REPO_ID}.",
              f"Request access at {HF_MODEL_URL}, then re-run this command.")
        sys.exit(1)
    except RepositoryNotFoundError:
        _loud("HUGGING FACE MODEL NOT FOUND",
              f"Expected repo: {HF_MODEL_URL}",
              "Check the repo id and re-run this command.")
        sys.exit(1)


def _download_pretrained_model(target_path, token):
    """Download the pretrained Forge model from Hugging Face and save it to target_path."""
    _loud("DOWNLOADING PRETRAINED MODEL FROM HUGGING FACE",
          f"Repo: {HF_REPO_ID}",
          f"File: {HF_MODEL_FILENAME}",
          f"Save: {target_path}")
    cached_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME, token=token)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copyfile(cached_path, target_path)
    _loud("DOWNLOAD COMPLETE",
          f"Model saved to {target_path}")


def _resolve_model_path(requested_path):
    """Return a local model path, downloading the pretrained model if none exists."""
    # 1. Local model at the requested path
    if os.path.isfile(requested_path):
        _loud("USING LOCAL MODEL", requested_path)
        return requested_path

    # 2. Fallback: search the repo's models directory
    models_dir_model = os.path.join(Constants.MODELS_DIR, Constants._FORGE_PKL_NAME)
    if os.path.isfile(models_dir_model):
        _loud("USING LOCAL MODEL (from models dir)", models_dir_model)
        return models_dir_model

    # 3. No local model: download the pretrained model from Hugging Face
    checked = [requested_path]
    if models_dir_model != requested_path:
        checked.append(models_dir_model)
    _loud("NO LOCAL MODEL FOUND", *[f"Checked {path}" for path in checked])
    token = _ensure_hf_token()
    _check_hf_access(token)
    _download_pretrained_model(models_dir_model, token)
    return models_dir_model


def main():
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default=DEFAULT_TRAIN_CONFIG_YAML,
                        help='Path to the training configuration YAML file')
    parser.add_argument('--input_forge_pkl', type=str, default=DEFAULT_INPUT_FORGE_PKL,
                        help='Path to pre-trained or fine-tuned Forge pickle file')
    parser.add_argument('--model_type', type=str, default=Constants.FORGE_PRE_TRAIN,
                        help=('The type of the pretrained model to load.'
                              'Available options: ' + ', '.join([Constants.FORGE_PRE_TRAIN,
                                                                 Constants.FORGE_FINE_TUNE_INTEGRAL_GAP,
                                                                 Constants.FORGE_FINE_TUNE_VARIABLE_PROBA])))
    parser.add_argument('--input_mips', type=str, default=DEFAULT_INPUT_MIPS,
                        help='Path to MIP file, directory, or model')
    parser.add_argument('--input_mip_instances_file', type=str, default=DEFAULT_INPUT_MIP_INSTANCES_FILE,
                        help='Directory containing input MIP instance files')
    parser.add_argument('--output_mip_to_embeddings_pkl', type=str, default=DEFAULT_OUTPUT_MIP_TO_EMBEDDINGS_PKL,
                        help='Output pickle file for embeddings')
    parser.add_argument('--instance_embedding_only', dest='instance_embedding_only', action='store_true',
                        help='Only save instance embedding')
    parser.add_argument('--no-instance-embedding-only', dest='instance_embedding_only', action='store_false',
                        help='Save instance, variable, and constraint embeddings')
    parser.set_defaults(instance_embedding_only=True)

    args = parser.parse_args()

    # Create Forge with its training configuration
    forge = Forge(args.train_config_yaml)

    # Resolve the model file: local if present, otherwise download from Hugging Face
    input_forge_pkl = _resolve_model_path(args.input_forge_pkl)

    # Generate embeddings
    mip_to_embeddings(forge=forge,
                      input_forge_pkl=input_forge_pkl,
                      model_type=args.model_type,
                      input_mips=args.input_mips,
                      input_mip_instances_file=args.input_mip_instances_file,
                      output_mip_to_embeddings_pkl=args.output_mip_to_embeddings_pkl,
                      instance_embedding_only=args.instance_embedding_only)


if __name__ == "__main__":
    main()
