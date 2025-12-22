import argparse

from forge.embeddings import Forge
from forge.pipeline import mip_to_embeddings
from forge.utils import Constants


if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/train_config.yaml',
                        help='Path to the training configuration YAML file')
    parser.add_argument('--input_forge_pkl', type=str, default='../models/forge_pretrained.pkl',
                        help='Path to pre-trained or fine-tuned Forge pickle file')
    parser.add_argument('--model_type', type=str, default=Constants.FORGE_PRE_TRAIN,
                        help=('The type of the pretrained model to load.'
                              'Available options: ' + ', '.join([Constants.FORGE_PRE_TRAIN,
                                                                 Constants.FORGE_FINE_TUNE_INTEGRAL_GAP,
                                                                 Constants.FORGE_FINE_TUNE_VARIABLE_PROBA])))
    parser.add_argument('--input_mips', type=str, default='../data/instances/',
                        help='Path to MIP file, directory, or model')
    parser.add_argument('--input_mip_instances_file', type=str, default='../data/configs/all.txt',
                        help='Directory containing input MIP instance files')
    parser.add_argument('--output_mip_to_embeddings_pkl', type=str,  default='../models/mip_to_embeddings.pkl',
                        help='Output pickle file for embeddings')
    parser.add_argument('--instance_embedding_only', dest='instance_embedding_only', action='store_true',
                        help='Only save instance embedding')
    parser.add_argument('--no-instance-embedding-only', dest='instance_embedding_only', action='store_false',
                        help='Save instance, variable, and constraint embeddings')
    parser.set_defaults(instance_embedding_only=True)

    args = parser.parse_args()

    # Create Forge with its training configuration
    forge = Forge(args.train_config_yaml)

    # Generate embeddings
    mip_to_embeddings_dict = mip_to_embeddings(forge=forge,
                                               input_forge_pkl=args.input_forge_pkl,
                                               model_type=args.model_type,
                                               input_mips=args.input_mips,
                                               input_mip_instances_file=args.input_mip_instances_file,
                                               output_mip_to_embeddings_pkl=args.output_mip_to_embeddings_pkl,
                                               instance_embedding_only=args.instance_embedding_only)
