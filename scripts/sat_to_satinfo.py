import argparse
import sys
import os

# Add parent directory to path to use local forge module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.embeddings import Forge
from forge.pipeline import sat_to_satinfo

if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_yaml', type=str, default='../forge/configs/train_config.yaml',
                        help='Path to the training configuration YAML file')
    parser.add_argument('--input_sat_folder', type=str, default='../data/satcomp2022anni_sat_instances/',
                        help='Directory containing input SAT instance files (LP/MPS format)')
    parser.add_argument('--input_sat_instances_file', type=str, default=None,
                        help='File containing list of SAT instances to use from input_sat_folder')
    parser.add_argument('--output_sat_to_satinfo_pkl', type=str,
                        default='../models/satcomp2022anni_sat_to_satinfo.pkl',
                        help='Output path for the sat_to_satinfo pickle')
    parser.add_argument('--num_parallel_workers', type=int, default=1,
                        help='The number of parallel workers to use for satinfo generation')
    parser.add_argument('--max_graph_nodes', type=int, default=100000,
                        help='Maximum number of graph nodes when converting SAT instances to bipartite graph')
    args = parser.parse_args()

    # Create Forge with training configuration (uses seed for SAT solver)
    forge = Forge(args.train_config_yaml)

    # Generate satinfo and save output pickle
    try:
        print(f"\n{'='*80}")
        print(f"Starting SAT to SATInfo conversion...")
        print(f"{'='*80}")
        print(f"Input folder: {args.input_sat_folder}")
        print(f"Input instances file: {args.input_sat_instances_file}")
        print(f"Output path: {args.output_sat_to_satinfo_pkl}")
        print(f"Number of parallel workers: {args.num_parallel_workers}")
        print(f"Max graph nodes: {args.max_graph_nodes}")
        print(f"{'='*80}\n")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(args.output_sat_to_satinfo_pkl))
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        print(f"Output directory exists: {os.path.isdir(output_dir)}\n")
        
        sat_to_satinfo_dict = sat_to_satinfo(forge=forge,
                                             input_sat_folder=args.input_sat_folder,
                                             input_sat_instances_file=args.input_sat_instances_file,
                                             output_sat_to_satinfo_pkl=args.output_sat_to_satinfo_pkl,
                                             num_parallel_workers=args.num_parallel_workers,
                                             has_return=True,
                                             max_graph_nodes=args.max_graph_nodes)
        
        num_instances = len(sat_to_satinfo_dict) if sat_to_satinfo_dict else 0
        print(f"✓ Successfully generated SATInfo for {num_instances} instances\n")
        
        # Verify output file exists and check size
        if os.path.exists(args.output_sat_to_satinfo_pkl):
            file_size_mb = os.path.getsize(args.output_sat_to_satinfo_pkl) / (1024 * 1024)
            print(f"{'='*80}")
            print(f"✓ OUTPUT FILE CREATED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"File: {args.output_sat_to_satinfo_pkl}")
            print(f"Size: {file_size_mb:.2f} MB")
            print(f"Instances stored: {num_instances}")
            print(f"{'='*80}\n")
        else:
            print(f"✗ WARNING: Output file was not created!")
            print(f"Expected path: {args.output_sat_to_satinfo_pkl}")
            exit(1)
            
    except Exception as e:
        print(f"\n✗ Error during SAT to SATInfo conversion: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
