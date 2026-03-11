import argparse
import sys
import os

# Add parent directory to path to use local forge module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.embeddings import Forge
from forge.pipeline import sat_to_embeddings
from forge.utils import Constants


if __name__ == "__main__":

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/train_config.yaml',
                        help='Path to the training configuration YAML file')
    parser.add_argument('--input_forge_pkl', type=str, default='../models/forge_sat_pretrained_subset_tester.pkl',
                        help='Path to pre-trained or fine-tuned Forge pickle file')
    parser.add_argument('--model_type', type=str, default=Constants.FORGE_PRE_TRAIN,
                        help=('The type of the pretrained model to load.'
                              'Available options: ' + ', '.join([Constants.FORGE_PRE_TRAIN,
                                                                 Constants.FORGE_FINE_TUNE_INTEGRAL_GAP,
                                                                 Constants.FORGE_FINE_TUNE_VARIABLE_PROBA])))
    parser.add_argument('--input_sat_folder', type=str, default='../data/g4satbench_sat_instances/',
                        help='Path to directory containing SAT instance files (LP/MPS format)')
    parser.add_argument('--input_sat_instances_file', type=str, default=None,
                        help='File containing list of SAT instances to use from input_sat_folder')
    parser.add_argument('--output_sat_to_embeddings_pkl', type=str,  default='../models/sat_to_embeddings_subset_tester.pkl',
                        help='Output pickle file for embeddings')
    parser.add_argument('--instance_embedding_only', dest='instance_embedding_only', action='store_true',
                        help='Only save instance embedding')
    parser.add_argument('--no-instance-embedding-only', dest='instance_embedding_only', action='store_false',
                        help='Save instance, variable, and clause embeddings')
    parser.add_argument('--max_graph_nodes', type=int, default=100000,
                        help='Maximum number of graph nodes when converting SAT instances to bipartite graph')
    parser.set_defaults(instance_embedding_only=True)

    args = parser.parse_args()

    # Create Forge with its training configuration
    forge = Forge(args.train_config_yaml)

    # Generate embeddings
    sat_to_embeddings_dict = sat_to_embeddings(forge=forge,
                                               input_forge_pkl=args.input_forge_pkl,
                                               model_type=args.model_type,
                                               input_sat_folder=args.input_sat_folder,
                                               input_sat_instances_file=args.input_sat_instances_file,
                                               output_sat_to_embeddings_pkl=args.output_sat_to_embeddings_pkl,
                                               instance_embedding_only=args.instance_embedding_only,
                                               max_graph_nodes=args.max_graph_nodes)