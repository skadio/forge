import argparse

from forge.embeddings import Forge
from forge.pipeline import mip_to_mipinfo

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/train_config.yaml',
                        help='Path to the training configuration YAML file')
    parser.add_argument('--input_mip_folder', type=str, default='../data/instances/',
                        help='Directory containing input MIP instance files')
    parser.add_argument('--input_mip_instances_file', type=str, default='../data/configs/pretrain.txt',
                        help='Directory containing input MIP instance files')
    parser.add_argument('--output_mip_to_mipinfo_pkl', type=str, default='../models/mip_to_mipinfo.pkl',
                        help='Output path for the mip_to_mipinfo pickle')
    parser.add_argument('--relaxation_list', nargs='*', type=float, default=[0.05, 0.01],
                        help='Space-separated list of relaxation values to use during pretraining')
    args = parser.parse_args()

    # Create Forge with training configuration (uses seed for mip solver)
    forge = Forge(args.train_config_yaml)

    # Generate mipinfo and save output pickle
    mip_to_mipinfo_dict = mip_to_mipinfo(forge=forge,
                                         input_mip_folder=args.input_mip_folder,
                                         input_mip_instances_file=args.input_mip_instances_file,
                                         output_mip_to_mipinfo_pkl=args.output_mip_to_mipinfo_pkl,
                                         relaxation_list=args.relaxation_list)
