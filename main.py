import argparse

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RPG', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=command_line_configs
    )
    pipeline.run()
