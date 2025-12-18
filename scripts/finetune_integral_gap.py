import argparse

from forge.embeddings import Forge
from forge.pipeline import finetune_integral_gap
from forge.utils import Constants


if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='/forge/configs/train_config.yaml',
                        help='Path to training config YAML file')
    parser.add_argument('--input_forge_pkl', type=str, default='/models/forge_pretrained.pkl',
                        help='Path to trained Forge pickle file')
    parser.add_argument('--input_mip_folder', type=str, default='/data/train/',
                        help='Path to MIP folder')
    parser.add_argument('--input_mip_instances_file', type=str, default='./data/pretrain.txt',
                        help='Directory containing input MIP instance files')
    parser.add_argument('--output_forge_finetuned_pkl', type=str, default='/models/forge_integral_gap.pkl',
                        help='Path to trained Forge pickle file')
    parser.add_argument('--output_mip_to_gapinfo_pkl', type=str,  default='/models/mip_to_gapinfo.pkl',
                        help='Output pickle file to store mip_to_integral_gap')
    parser.add_argument('--input_mip_to_gapinfo_pkl', type=str, default=None,
                        help='Optional path to an existing input_mip_to_gapinfo_pkl to load instead of creating')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_instance', type=int, default=10,
                        help='Number of training steps per MIP instance per epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    # TODO default weight decay for pretraining is 1e-4 and here it is 5e-4, it's different, yes?
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for the optimizer')
    parser.add_argument('--max_graph_nodes', type=int, default=30000,
                        help='Maximum number of graph nodes when converting MIP instances to bipartite graph')
    parser.add_argument('--gapinfo_time_limit', type=int, default=120,
                        help='Time limit in seconds for computing integral gap info (default: 120)')
    parser.add_argument('--num_parallel_workers', type=int, default=5,
                        help='Number of parallel workers to use for processing MIP instances')
    args = parser.parse_args()

    # Forge model ready for fine-tuning
    forge = Forge(args.train_config_yaml)

    # Fine-tune Forge to predict integral gaps
    finetune_integral_gap(forge=forge,
                          input_forge_pkl=args.input_forge_pkl,
                          input_mip_instances_file=args.input_mip_instances_file,
                          model_type=Constants.FORGE_FINE_TUNE_INTEGRAL_GAP,
                          input_mip_folder=args.input_mip_folder,
                          output_forge_finetuned_pkl=args.output_forge_finetuned,
                          output_mip_to_gapinfo_pkl=args.output_mip_to_gapinfo_pkl,
                          input_mip_to_gapinfo_pkl=args.input_mip_to_gapinfo_pkl,
                          epochs=args.epochs,
                          steps_per_instance=args.steps_per_instance,
                          learning_rate=args.learning_rate,
                          weight_decay=args.weight_decay,
                          max_graph_nodes=args.max_graph_nodes,
                          gapinfo_time_limit=args.gapinfo_time_limit, 
                          num_parallel_workers=args.num_parallel_workers)
