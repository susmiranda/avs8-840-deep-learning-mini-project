#!/bin/sh
python make_chunks_manifests.py --dataset_path ~/datasets/covid19_cough --train_csv ~/datasets/covid19_cough/metadata/train.csv \
--val_csv ~/datasets/covid19_cough/metadata/validation.csv --test_csv ~/datasets/covid19_cough/metadata/test.csv \
--nl_csv ~/datasets/covid19_cough/no_labeled.csv --train_chunks_dir ~/datasets/covid19_cough/train_chunks \
--val_chunks_dir ~/datasets/covid19_cough/val_chunks --test_chunks_dir ~/datasets/covid19_cough/test_chunks \
--nl_chunks_dir ~/datasets/covid19_cough/unlabeled_chunks --output_dir ~/datasets/covid19_cough/manifests --opsys linux