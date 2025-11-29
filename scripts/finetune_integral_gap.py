import argparse

from forge.embeddings import Forge
from forge.pipeline import finetune_integral_gap
from forge.utils import Constants


if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='/forge/configs/train_config.yaml',
                        help='Path to training config YAML file')
    parser.add_argument('--input_forge_pretrained.pkl', type=str, default='/models/forge_pretrained.pkl',
                        help='Path to trained Forge pickle file')
    parser.add_argument('--input_mip_folder', type=str, default='/data/train/',
                        help='Path to MIP folder')
    parser.add_argument('--output_forge_finetuned.pkl', type=str, default='/models/forge_integral_gap.pkl',
                        help='Path to trained Forge pickle file')
    parser.add_argument('--output_mip_to_gapinfo_pkl', type=str,  default='/models/mip_to_gapinfo.pkl',
                        help='Output pickle file to store mip_to_integral_gap')
    parser.add_argument('--input_mip_to_integral_gap_pkl', type=str, default=None,
                        help='Optional path to an existing output_mip_to_integral_gap_pkl to load instead of creatig')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_instance', type=int, default=10,
                        help='Number of training steps per MIP instance per epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    # TODO default max nodes for pretraining is 2100 and here it is 30000, very different, no?
    parser.add_argument('--max_dgl_nodes', type=int, default=30000,
                        help='Maximum number of graph nodes when converting MIP instances to DGL graphs')
    args = parser.parse_args()

    # Forge model ready for fine-tuning
    forge = Forge(args.train_config_yaml)

    # Load pre-trained Forge model ready for fine-tuning
    forge.load_model(input_forge_pkl=args.input_forge_pretrained, model_type=Constants.FORGE_FINE_TUNE_INTEGRAL_GAP)

    # Fine-tune Forge to predict integral gaps
    finetune_integral_gap(forge=forge,
                          input_mip_folder=args.input_mip_folder,
                          output_forge_finetuned_pkl=args.output_forge_finetuned,
                          output_mip_to_gapinfo_pkl=args.output_mip_to_gapinfo_pkl,
                          epochs=args.epochs,
                          steps_per_instance=args.steps_per_instance,
                          learning_rate=args.learning_rate,
                          max_dgl_nodes=args.max_dgl_nodes)
