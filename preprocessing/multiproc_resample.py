# Taken from (https://github.com/SarthakYadav)
# MIT License
# Copyright (c) 2021 Sarthak Yadav

"""
Preprocessing Step 1: Convert audio files into 24000 Hz audio   // changed the sample rate from 22050 Hz
Also, converts webm files to wav
"""

import argparse
import glob
import os
import subprocess as sp
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.description = "Multi-processing resampling script"
parser.add_argument(
    "--src_path", type=str, help="path to source directory containing audio files"
)
parser.add_argument(
    "--dst_path",
    type=str,
    help="path to destination directory where resampled files will be stored",
)
parser.add_argument("--sample_rate", type=int, default=24000, help="target sample rate")
parser.add_argument(
    "--opsys",
    type=str,
    default="linux",
    choices=["windows", "linux"],
    help="Operating system the code is ran on.",
)

args = parser.parse_args()

files_webm = glob.glob(os.path.join(args.src_path, "*.webm"))
files_ogg = glob.glob(os.path.join(args.src_path, "*.ogg"))
files = files_ogg + files_webm
#print(len(files))
tgt_dir = args.dst_path
if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

lf = len(files)
SAMPLE_RATE = args.sample_rate


def process_idx(idx):
    f = files[idx]
    if args.opsys == "windows":
        fname = f.split("\\")[-1].split(".")[0]
    else:
        fname = f.split("/")[-1].split(".")[0]

    tgt_path = os.path.join(tgt_dir, fname + ".wav")
    if args.opsys == "windows":
        command = "ffmpeg -loglevel 0 -nostats -i {} -ac 1 -ar {} {}".format(
            f, SAMPLE_RATE, tgt_path
        )
    else:
        command = "ffmpeg -loglevel 0 -nostats -i '{}' -ac 1 -ar {} '{}'".format(
            f, SAMPLE_RATE, tgt_path
        )
    sp.call(command, shell=True)
    if idx % 500 == 0:
        print("Done: {:04d}/{}".format(idx, lf))


if __name__ == "__main__":
    pool = Pool(6)
    o = pool.map_async(process_idx, range(lf))
    res = o.get()
    pool.close()
    pool.join()
