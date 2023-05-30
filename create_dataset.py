import argparse
import json

from ann_benchmarks.datasets import DATASETS, get_dataset_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--additional-args", type=str, required=False)
    args = parser.parse_args()
    fn = get_dataset_fn(args.dataset)
    if args.additional_args:
        parameters = json.loads(args.additional_args)
        parameters['out_fn'] = fn
        DATASETS[args.dataset](**parameters)
    else:
        DATASETS[args.dataset](fn)
