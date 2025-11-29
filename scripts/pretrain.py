import argparse

from forge.embeddings import Forge
from forge.pipeline import pretrain

if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='/forge/configs/train_config.yaml',
                        help='Path to the training configuration YAML file')
    parser.add_argument('--input_mip_folder', type=str, default='/data/train/',
                        help='Directory containing input MIP instance files')
    parser.add_argument('--output_mip_to_mipinfo_pkl', type=str, default='/models/mip_to_mipinfo.pkl',
                        help='Output path for the mip_to_mipinfo pickle')
    parser.add_argument('--output_forge_pretrained_pkl', type=str, default='/models/forge_pretrained.pkl',
                        help='Output path for the pretrained Forge pickle')
    parser.add_argument('--output_log_file', type=str, default='/models/forge_pretrained.log',
                        help='Path to write the pretraining log')
    parser.add_argument('--input_mip_to_mipinfo_pkl', type=str, default=None,
                        help='Optional path to an existing mip_to_mipinfo pickle to load instead of generating it')
    parser.add_argument('--relaxation_list', nargs='+', type=float, default=[0.05, 0.01],
                        help='Space-separated list of relaxation values to use during pretraining')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_instance', type=int, default=10,
                        help='Number of training steps per MIP instance per epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--max_dgl_nodes', type=int, default=2100,
                        help='Maximum number of graph nodes when converting MIP instances to DGL graphs')
    args = parser.parse_args()

    # Create Forge with training configuration
    forge = Forge(args.train_config_yaml)

    # Pre-train Forge
    pretrain(forge=forge,
             input_mip_folder=args.input_mip_folder,
             output_mip_to_mipinfo_pkl=args.output_mip_to_mipinfo_pkl,
             output_forge_pretrained_pkl=args.output_forge_pretrained_pkl,
             output_log_file=args.output_log_file,
             input_mip_to_mipinfo_pkl=args.input_mip_to_mipinfo_pkl,
             relaxation_list=args.relaxation_list,
             epochs=args.epochs,
             steps_per_instance=args.steps_per_instance,
             learning_rate=args.learning_rate,
             max_dgl_nodes=args.max_dgl_nodes)

