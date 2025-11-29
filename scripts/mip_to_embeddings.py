import argparse
import pickle

from forge.embeddings import Forge
from forge.pipeline import mip_to_embeddings
from forge.utils import Constants


def run(train_config_yaml, input_forge_pkl, input_mips, output_mip_to_embeddings_pkl):
    # Create Forge with its training configuration
    forge = Forge(train_config_yaml)

    # Load pre-trained Forge model
    forge.load_model(input_forge_pkl=input_forge_pkl, model_type=Constants.FORGE_PRE_TRAINED)

    # Generate embeddings
    mip_to_embeddings_dict = mip_to_embeddings(forge=forge,
                                               input_mips=input_mips,
                                               output_mip_to_embeddings_pkl=output_mip_to_embeddings_pkl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='/forge/configs/train_config.yaml')
    parser.add_argument('--input_forge_pkl', type=str, required=True, help='Path to trained Forge pickle file')
    parser.add_argument('--input_mips', type=str, required=True, help='Path to MIP file, directory, or model')
    parser.add_argument('--output_mip_to_embeddings_pkl', type=str, required=True,
                        help='Output pickle file for embeddings')
    args = parser.parse_args()

    run(train_config_yaml=args.train_config_yaml,
        input_forge_pkl=args.input_forge_pkl,
        input_mips=args.input_mips,
        output_mip_to_embeddings_pkl=args.output_mip_to_embeddings_pkl)
