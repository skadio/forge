import argparse

from forge.embeddings import Forge
from forge.pipeline import mip_to_gapinfo
from forge.utils import Constants

if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='/forge/configs/train_config.yaml',
                        help='Path to the training configuration YAML file')
    parser.add_argument('--input_forge_pkl', type=str, default='/models/forge_pretrained.pkl',
                        help='Path to pre-trained or fine-tuned Forge pickle file')
    parser.add_argument('--input_mips', type=str, default='/data/train/',
                        help='Path to MIP file, directory, or model')
    parser.add_argument('--output_mip_to_gap_info_pkl', type=str, default='/models/output_mip_to_gap_info.pkl',
                        help='Output pickle file for gap info')
    # TODO consider removing problem_type in future
    parser.add_argument('--problem_type', type=str, default='SC',
                        help='The type of the problem domain CA, GISP, MVC, SC')

    args = parser.parse_args()

    # Create Forge with its training configuration
    forge = Forge(args.train_config_yaml)

    # Generate embeddings
    mip_to_gap_info_dict = mip_to_gapinfo(forge=forge,
                                          input_forge_pkl=args.input_forge_pkl,
                                          model_type=Constants.FORGE_FINE_TUNE_INTEGRAL_GAP,
                                          input_mips=args.input_mips,
                                          output_mip_to_gap_info_pkl=args.output_mip_to_gap_info_pkl,
                                          problem_type=args.problem_type)
