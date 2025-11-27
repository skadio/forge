from forge.embeddings import Forge
from forge.processor import DataProcessor

import argparse


def run(train_config_file_path, train_config_version, input_folder, output_mip_info_file, output_forge_file, perturb_list):

    # Process MIP instances to MIPInfo Objects
    dp = DataProcessor()
    dp.convert_mip_to_mipinfo(input_folder=input_folder,
                              output_file=output_file,
                              perturb_list=perturb_list,
                              has_return=False)

    train_list = dp.get_train_list(['.data/intermediate_files/mips_to_dgl.pkl'])

    # Unsupervised Pre-Training
    forge = Forge(train_config_file_path, train_config_version)
    forge.pretrain(output_file,
                   train_list=train_list,
                   epochs=10,
                   steps_per_instance=10,
                   lr=1e-4,
                   log_path='./data/log/unsupervised_train_log.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_file_path', type=str, default='../forge/train_config.yaml')
    parser.add_argument('--train_config_version', type=str, default='default')
    parser.add_argument('--input_folder', type=str, default='../data/train/')
    parser.add_argument('--output_mip_info_file', type=str, default='../models/mip_to_mipinfo.pkl')
    parser.add_argument('--output_forge_file', type=str, default='../models/forge_pretrained.pkl')
    parser.add_argument('--perturb_list', default=[0.05, 0.01])
    args = parser.parse_args()

    # For an interactive example run, please see 'pretrain.ipynb'.
    # Ensure instances are placed in data/train
    run(train_config_file_path=args.train_config_file_path,
        train_config_version=args.train_config_version,
        input_folder=args.input_folder,
        output_mip_info_file=args.output_mip_info_file,
        output_forge_file=args.output_forge_file,
        perturb_list=args.perturb_list)
