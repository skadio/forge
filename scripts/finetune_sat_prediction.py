import argparse
import sys
import os

# Add parent directory to path to use local forge module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.embeddings import Forge
from forge.pipeline import finetune_sat_prediction

if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/train_config.yaml',
                        help='Path to training config YAML file')
    parser.add_argument('--input_forge_pkl', type=str, default='../models/forge_sat_pretrained.pkl',
                        help='Path to pretrained Forge pickle file (from sat_pretrain.py)')
    parser.add_argument('--input_sat_folder', type=str, default='../data/subset_sat_instances/',
                        help='Path to SAT folder containing SAT instances in LP/MPS format')
    parser.add_argument('--input_sat_instances_file', type=str, default=None,
                        help='File containing list of SAT instances to use from input_sat_folder')
    parser.add_argument('--output_forge_finetuned_pkl', type=str, default='../models/forge_sat_finetuned.pkl',
                        help='Path to save fine-tuned Forge pickle file')
    parser.add_argument('--output_sat_to_satinfo_pkl', type=str, default='../models/sat_to_satinfo.pkl',
                        help='Output pickle file to store sat_to_satisfiability_info mapping')
    parser.add_argument('--input_sat_to_satinfo_pkl', type=str, default=None,
                        help='Optional path to an existing sat_to_satinfo_pkl to load instead of creating')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_instance', type=int, default=10,
                        help='Number of training steps per SAT instance per epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for the optimizer')
    parser.add_argument('--max_graph_nodes', type=int, default=100000,
                        help='Maximum number of graph nodes (clauses + variables) when converting SAT instances to bipartite graph')
    args = parser.parse_args()

    # Forge model ready for fine-tuning
    forge = Forge(args.train_config_yaml)

    # Print GPU information
    import torch
    print(f"\n{'='*80}")
    print(f"GPU INFORMATION")
    print(f"{'='*80}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"Forge device: {forge.device}", flush=True)
    else:
        print(f"WARNING: CUDA is not available - using CPU", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Fine-tune Forge to predict SAT satisfiability
    finetune_sat_prediction(forge=forge,
                           input_forge_pkl=args.input_forge_pkl,
                           model_type='fine-tune',
                           input_sat_folder=args.input_sat_folder,
                           input_sat_instances_file=args.input_sat_instances_file,
                           output_forge_finetuned_pkl=args.output_forge_finetuned_pkl,
                           output_sat_to_satinfo_pkl=args.output_sat_to_satinfo_pkl,
                           input_sat_to_satinfo_pkl=args.input_sat_to_satinfo_pkl,
                           epochs=args.epochs,
                           steps_per_instance=args.steps_per_instance,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           max_graph_nodes=args.max_graph_nodes)

