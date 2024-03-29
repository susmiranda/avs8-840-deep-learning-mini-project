import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument('--manifest_path', type=str, required=True, help='Path to the manifest file')
args = ap.parse_args()

parent_dir = os.path.dirname(args.manifest_path)
df = pd.read_csv(args.manifest_path)
filenames = df['files'].values
filenames = [os.path.basename(filename) for filename in filenames]
filenames = [filename.split('_')[0] for filename in filenames]
filenames = np.unique(filenames)
train, val = train_test_split(filenames, train_size=0.8, random_state=42)
train_df = pd.DataFrame(columns=['files'])
val_df = pd.DataFrame(columns=['files'])
for file in train:
    train_df = pd.concat((train_df, df[df['files'].str.contains(file)]), ignore_index=True)
for file in val:
    val_df = pd.concat((val_df, df[df['files'].str.contains(file)]), ignore_index=True)
train_df.to_csv(os.path.join(parent_dir, 'unlabeled_train.csv'), index=False)
val_df.to_csv(os.path.join(parent_dir, 'unlabeled_val.csv'), index=False)