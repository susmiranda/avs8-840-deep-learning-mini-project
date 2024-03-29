# Taken from (https://github.com/SarthakYadav)
# MIT License
# Copyright (c) 2021 Sarthak Yadav

"""
Preprocessing Step 3: Make Manifest csv files from chunks for training validation and test
    This is the final step
    
    Modified it to be able to work with Covid-19 Cough Audio Classification dataset(https://www.kaggle.com/datasets/andrewmvd/covid19-cough-audio-classification)
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import tqdm

# python make_chunks_manifests.py --dataset_path dataset/valid_dataset/ --train_csv metadata/train.csv --val_csv metadata/validation.csv --test_csv metadata/test.csv --train_chunks_dir train_24000_chunks --val_chunks_dir validation_24000_chunks --test_chunks_dir test_24000_chunks --output_dir chunks_meta_dir --nl_csv  metadata/no_labeled.csv --nl_chunks_dir no_labeled_24000_chunks
parser = argparse.ArgumentParser()
parser.description = "Script to make experiment manifests"
parser.add_argument(
    "--dataset_path",
    type=str,
    default=None,
    help="path to dataset parent folder",
)

parser.add_argument(
    "--train_csv",
    type=str,
    default=None,
    help="path to train.csv file found in metadata",
)
parser.add_argument(
    "--val_csv",
    type=str,
    default=None,
    help="path to validation.csv file found in metadata",
)
parser.add_argument(
    "--test_csv",
    type=str,
    default=None,
    help="path to test.csv file found in metadata",
)
parser.add_argument(
    "--nl_csv",
    type=str,
    default=None,
    help="path to no_labeled.csv file found in metadata",
)

parser.add_argument(
    "--train_chunks_dir", type=str, help="path to directory containing train chunks"
)
parser.add_argument(
    "--val_chunks_dir",
    type=str,
    default=None,
    help="path to directory containing val chunks",
)
parser.add_argument(
    "--test_chunks_dir",
    type=str,
    default=None,
    help="path to directory containing test chunks",
)
parser.add_argument(
    "--nl_chunks_dir",
    type=str,
    default=None,
    help="path to directory containing no_labeled chunks",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="path to directory for storing manifest csv files used for training",
)
parser.add_argument(
    "--opsys",
    type=str,
    choices=["windows", "linux"],
    default="linux",
    help="Operating system the code is run on.",
)


def process_set(csv_path, chuck_path, meta_name, opsys):

    setdf = pd.read_csv(csv_path)
    set_files = setdf["uuid"].values
    set_labels = setdf["status"].values
    set_chunks_dir = chuck_path
    set_lbl_map = {}
    for i in tqdm.tqdm(range(len(set_files))):
        ext = set_files[i]
        lbl = set_labels[i]
        # ext = f.split("/")[-1].split(".")[0]
        set_lbl_map[str(ext)] = lbl

    chunk_labels = []
    chunk_exts = []
    chunk_files = []

    set_chunk_files = glob.glob(os.path.join(set_chunks_dir, "*.wav"))
    set_keys = list(set_lbl_map.keys())

    for f in tqdm.tqdm(set_chunk_files):
        if opsys == "windows":
            ext = f.split("\\")[-1].split(".")[0].split("_")[0]
        else:
            ext = f.split("/")[-1].split(".")[0].split("_")[0]
        chunk_labels.append(set_lbl_map[ext])
        chunk_exts.append(ext)
        chunk_files.append(f)

    chunk_labels = np.asarray(chunk_labels)
    chunk_exts = np.asarray(chunk_exts)
    chunk_files = np.asarray(chunk_files)

    set_chunk = pd.DataFrame()
    set_chunk["files"] = chunk_files
    set_chunk["labels"] = chunk_labels
    set_chunk["ext"] = chunk_exts
    set_chunk = set_chunk.iloc[np.random.permutation(len(set_chunk))]

    set_chunk.to_csv(os.path.join(args.output_dir, meta_name), index=False)

    vocab = np.unique(chunk_labels)
    lbl_map = {}
    for inx in range(len(vocab)):
        lbl_map[vocab[inx]] = inx
    with open(os.path.join(args.output_dir, "lbl_map.json"), "w") as fd:
        json.dump(lbl_map, fd)


def process_set_nolabels(csv_path, chuck_path, meta_name):

    # setdf = pd.read_csv(csv_path)
    # set_files = setdf["uuid"].values
    set_chunks_dir = chuck_path
    chunk_files = []

    set_chunk_files = glob.glob(os.path.join(set_chunks_dir, "*.wav"))
    for f in tqdm.tqdm(set_chunk_files):
        chunk_files.append(f)

    chunk_files = np.asarray(chunk_files)

    set_chunk = pd.DataFrame()
    set_chunk["files"] = chunk_files

    set_chunk = set_chunk.iloc[np.random.permutation(len(set_chunk))]

    set_chunk.to_csv(os.path.join(args.output_dir, meta_name), index=False)


if __name__ == "__main__":
    args = parser.parse_args()

    # add parent folder to paths
    args.train_csv = os.path.join(args.dataset_path, args.train_csv)
    args.test_csv = os.path.join(args.dataset_path, args.test_csv)
    args.val_csv = os.path.join(args.dataset_path, args.val_csv)
    args.train_chunks_dir = os.path.join(args.dataset_path, args.train_chunks_dir)
    args.val_chunks_dir = os.path.join(args.dataset_path, args.val_chunks_dir)
    args.test_chunks_dir = os.path.join(args.dataset_path, args.test_chunks_dir)
    args.output_dir = os.path.join(args.dataset_path, args.output_dir)
    args.nl_csv = os.path.join(args.dataset_path, args.nl_csv)
    args.nl_chunks_dir = os.path.join(args.dataset_path, args.nl_chunks_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_train = False if args.train_csv is None else True
    process_val = False if args.val_csv is None else True
    process_test = False if args.train_csv is None else True

    if process_train and process_val and process_test:
        process_set_nolabels(args.nl_csv, args.nl_chunks_dir, "no_labeled_chunk.csv")
        process_set(
            args.train_csv, args.train_chunks_dir, "train_chunk.csv", args.opsys
        )
        process_set(args.val_csv, args.val_chunks_dir, "val_chunk.csv", args.opsys)
        process_set(args.test_csv, args.test_chunks_dir, "test_chunk.csv", args.opsys)
