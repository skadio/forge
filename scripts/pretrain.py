import argparse

from forge.embeddings import Forge
from forge.pipeline import pretrain


def run(train_config_yaml, input_mip_folder, relaxation_list,
        output_mip_to_mipinfo_pkl, output_forge_pkl, output_log_file):

    # Create Forge with training configuration
    forge = Forge(train_config_yaml)

    # Pre-train Forge
    pretrain(forge,
             input_mip_folder, relaxation_list,
             output_mip_to_mipinfo_pkl, output_forge_pkl, output_log_file)


if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='/forge/configs/train_config.yaml')
    parser.add_argument('--input_mip_folder', type=str, default='/data/train/')
    parser.add_argument('--relaxation_list', nargs='+', type=float, default=[0.05, 0.01])
    parser.add_argument('--output_mip_to_mipinfo_pkl', type=str, default='/models/mip_to_mipinfo.pkl')
    parser.add_argument('--output_forge_pkl', type=str, default='/models/forge_pretrained.pkl')
    parser.add_argument('--output_log_file', type=str, default='/models/forge_pretrained.log')
    args = parser.parse_args()

    # Run
    run(train_config_yaml=args.train_config_yaml,
        input_mip_folder=args.input_mip_folder,
        relaxation_list=args.relaxation_list,
        output_mip_to_mipinfo_pkl=args.output_mip_to_mipinfo_pkl,
        output_forge_pkl=args.output_forge_pkl,
        output_log_file=args.output_log_file)
