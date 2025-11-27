import os
import argparse
from forge.pipeline import pretrain


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--features", type=str, default=os.environ["SM_CHANNEL_FEATURES"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--config", type=str, default="train.yaml")

    args = parser.parse_args()
    args = vars(args)

    config_args = train_config()
    config_args['data'] = os.path.join(args['train'], config_args['data'])
    config_args['user_features'] = os.path.join(args['features'], config_args['user_features'])
    config_args['save_file'] = os.path.join(args['model_dir'], config_args['save_file'])

    print(">>> Training started")
    train(**config_args)
    print("<<< Training finished")