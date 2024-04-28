import argparse
import json
import random
from dataclasses import dataclass

import boto3
import objaverse
import tyro
from tqdm import tqdm


@dataclass
class Args:
    count: int
    """total number of files uploaded"""

    skip_completed: bool = False
    """whether to skip the files that have already been downloaded"""

    save_json_path: str = "."
    "where to save the json files"

    seed: int = 42


def get_completed_uids():
    # get all the files in the objaverse-images bucket
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("objaverse-images")
    bucket_files = [obj.key for obj in tqdm(bucket.objects.all())]

    dir_counts = {}
    for file in bucket_files:
        d = file.split("/")[0]
        dir_counts[d] = dir_counts.get(d, 0) + 1

    # get the directories with 12 files
    dirs = [d for d, c in dir_counts.items() if c == 12]
    return set(dirs)


# set the random seed to 42
if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)

    annotations = objaverse.load_annotations()

    # sort by number of likes
    sorted_annotations = sorted(annotations.items(), key=lambda item: item[1]['likeCount'], reverse=True)

    limited_sorted_annotations = sorted_annotations[:args.count]

    uid_to_name = dict()
    for uid, metadata in limited_sorted_annotations:
        uid_to_name[uid] = metadata['name']

    object_paths = objaverse._load_object_paths()

    uid_object_paths = [
        f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths[uid]}"
        for uid in uid_to_name.keys() 
    ]

    with open(f"{args.save_json_path}/input_models_path.json", "w") as f:
        json.dump(uid_object_paths, f, indent=2)
    
    with open(f"{args.save_json_path}/uid_to_name.json", "w") as f:
        json.dump(uid_to_name, f, indent=2)
