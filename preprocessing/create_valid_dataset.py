import argparse
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.description = "Multi-processing resampling script"
parser.add_argument(
    "--src_path", type=str, help="path to source directory containing .wav files"
)
parser.add_argument(
    "--dst_path",
    type=str,
    help="path to destination directory where resampled files will be stored",
)
parser.add_argument("--sample_rate", type=int, default=22050, help="target sample rate")

args = parser.parse_args()

if not os.path.exists(args.dst_path):
    os.makedirs(args.dst_path)

metadata = args.dst_path + "metadata/"
if not os.path.exists(metadata):
    os.makedirs(metadata)

df = pd.read_csv(args.src_path + "metadata_compiled.csv")

# cough_detected threshold = 0.8  and valid status labels, so it filters the empty samples
valid_samples = df[(df["cough_detected"] > 0.8)]
valid_samples_path = metadata + "valid_metadata_compiled.csv"
valid_samples.to_csv(valid_samples_path, index=False)

labeled_samples = valid_samples[valid_samples["status"].notnull()]
no_labeled_files = valid_samples[valid_samples["status"].isnull()]

labeled_audio_files = labeled_samples["uuid"].values
# Label ['healthy' 'symptomatic' 'COVID-19']     np.unique(labels)
labels = labeled_samples["status"].values

no_labeled_audio_files = no_labeled_files["uuid"].values

# Train 70% Validation 10% and Test 20%
x_train, x_test, y_train, y_test = train_test_split(
    labeled_audio_files, labels, test_size=0.2, random_state=42, stratify=labels
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)


def move_audio_files(set, folder_name):
    for filename in set:
        source_file = next(
            (
                f
                for f in os.listdir(args.src_path)
                if (f == filename + ".webm") or (f == filename + ".ogg")
            ),
            None,
        )
        # If the source file exists, move it to the destination directory
        if source_file:
            # Construct source and destination paths
            source = os.path.join(args.src_path, source_file)

            if not os.path.exists(args.dst_path + folder_name):
                os.makedirs(args.dst_path + folder_name)

            destination = os.path.join(args.dst_path + folder_name, source_file)

            # Move the file to the destination directory
            # shutil.move(source, destination)
            print(f"Moved {filename} to {destination}")
        else:
            print(f"File {filename} not found in {args.src_path}")

    set_metadata = valid_samples[valid_samples["uuid"].isin(set)]
    set_metadata.to_csv(metadata + folder_name + ".csv", index=False)


move_audio_files(x_train, "train")
move_audio_files(x_val, "validation")
move_audio_files(x_test, "test")
move_audio_files(no_labeled_audio_files, "no_labeled")
