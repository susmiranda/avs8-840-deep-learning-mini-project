# Taken from (https://github.com/SarthakYadav)
# MIT License
# Copyright (c) 2021 Sarthak Yadav


"""
Spectrogram Dataset modified it to be able to work with Covid-19 Cough Audio Classification dataset(https://www.kaggle.com/datasets/andrewmvd/covid19-cough-audio-classification)

"""

import glob
import json
import os
from typing import Optional, Tuple

import pandas as pd
import torch
import tqdm
import random
from torch.utils.data import Dataset
from torchaudio.transforms import TimeMasking, FrequencyMasking

from src.data.audio_parser import AudioParser
from src.data.utils import load_audio


class SpectrogramDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        labels_map: str,
        audio_config: dict,
        mode: Optional[str] = "multilabel",
        augment: Optional[bool] = False,
        labels_delimiter: Optional[str] = ",",
        extra_labels: Optional[str] = None,
        mask_param: Optional[int] = 15
    ) -> None:
        super(SpectrogramDataset, self).__init__()

        assert audio_config is not None
        df = pd.read_csv(manifest_path)
        self.files = df["files"].values 
        # Set labels_map to None for self supervised.

        #self.labels is populated with the labels extracted from the CSV file. 
        #If labels_map is provided, it reads the labels from the CSV file and maps them according to the labels_map
        if labels_map:
            assert os.path.isfile(labels_map)
            assert os.path.splitext(labels_map)[-1] == ".json"
            self.labels_delim = labels_delimiter
            with open(labels_map, "r") as fd:
                self.labels_map = json.load(fd)
            self.labels = df["labels"].values
            assert len(self.files) == len(self.labels)
        else:
            self.labels = None

        self.len = len(self.files)
        self.sr = audio_config.get("sample_rate", "22050")
        self.n_fft = audio_config.get("n_fft", 511)
        win_len = audio_config.get("win_len", None)
        self.extra_labels = []
        if not win_len:
            self.win_len = self.n_fft
        else:
            self.win_len = win_len
        hop_len = audio_config.get("hop_len", None)
        if not hop_len:
            self.hop_len = self.n_fft // 2
        else:
            self.hop_len = hop_len

        self.normalize = audio_config.get("normalize", True)
        self.augment = augment
        if self.augment:
            self.mask_param = mask_param
            random.seed(42)
            torch.manual_seed(42)
            self.time_mask = TimeMasking(time_mask_param=self.mask_param, iid_masks=True)
            self.freq_mask = FrequencyMasking(freq_mask_param=self.mask_param, iid_masks=True)
        self.min_duration = audio_config.get("min_duration", None)
        self.background_noise_path = audio_config.get("bg_files", None)
        if self.background_noise_path is not None:
            if os.path.exists(self.background_noise_path):
                self.bg_files = glob.glob(
                    os.path.join(self.background_noise_path, "*.wav")
                )
        else:
            self.bg_files = None

        self.mode = mode
        feature = audio_config.get("feature", "spectrogram")
        self.spec_parser = AudioParser(
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop_len,
            feature=feature,
        )
        if self.bg_files is not None:
            print("prepping bg_features")
            self.bg_features = []
            for f in tqdm.tqdm(self.bg_files):
                preprocessed_audio = self.__get_audio__(f)
                real, comp = self.__get_feature__(preprocessed_audio)
                self.bg_features.append(real)
        else:
            self.bg_features = None
        self.prefetched_labels = None
        if self.mode == "multilabel":
            self.fetch_labels()

        if extra_labels != None:
            extra_lbl = pd.read_csv(extra_labels)
            self.extra_labels[extra_labels["age"]]

        if labels_map:
            self.weighted_sampler = self.calc_weights()

    def masking(self, spec):
        for mask in range(2):
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)
        return spec
    
    def transform(self, spec):
        spec = torch.from_numpy(spec)
        spec = spec.expand(1, -1, -1)
        return spec

    def fetch_labels(self):
        self.prefetched_labels = []
        for lbl in tqdm.tqdm(self.labels):
            proc_lbl = self.__parse_labels__(lbl)
            self.prefetched_labels.append(proc_lbl.unsqueeze(0))
        self.prefetched_labels = torch.cat(self.prefetched_labels, dim=0)
        print(self.prefetched_labels.shape)

    def __get_audio__(self, f):
        audio = load_audio(f, self.sr, self.min_duration)
        return audio

    def __get_feature__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp = self.spec_parser(audio)
        return real, comp

    def get_bg_feature(self, index: int) -> torch.Tensor:
        if self.bg_files is None:
            return None
        real = self.bg_features[index]
        return real

    def __get_item_helper__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.files[index]
        preprocessed_audio = self.__get_audio__(f)
        real, comp = self.__get_feature__(preprocessed_audio)
        if self.labels is not None:
            lbls = self.labels[index]
            label_tensor = self.__parse_labels__(lbls)
            return real, comp, label_tensor
        else:
            return real, comp

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.labels is not None:
            real, comp, label_tensor = self.__get_item_helper__(index)
            real = self.transform(real)
            # Apply data augmentations
            if self.augment:
                real = self.masking(real)

            return real, label_tensor
        else:
            real, comp = self.__get_item_helper__(index)
            real = self.transform(real)
            return real

    def __parse_labels__(self, lbls: str) -> torch.Tensor:
        if self.mode == "multilabel":
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delim):
                label_tensor[self.labels_map[lbl]] = 1

            return label_tensor
        elif self.mode == "multiclass":
            return self.labels_map[lbls]

    def __len__(self):
        return self.len

    def get_bg_len(self):
        return len(self.bg_files)
    
    def calc_weights(self):
        from torch.utils.data import WeightedRandomSampler
        # Initialize an empty array for weights
        weights = []
        
        # Calculate label counts
        label_counts = {}
        for labels in self.labels:
            for label in labels.split(self.labels_delim):
                label_counts[label] = label_counts.get(label, 0) + 1
                
        for labels in self.labels:
            weight = sum(1.0 / label_counts[label] for label in labels.split(self.labels_delim))
            weights.append(weight)
        
        weights = torch.FloatTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler
            
class SpecFeatDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        metadata_path: str,
        labels_map: str,
        audio_config: dict,
        augment: bool,
        mask_param: int = 15, 

        ) -> None:
        super(SpecFeatDataset, self).__init__()

        assert manifest_path is not None, 'Manifest path is required.'
        assert metadata_path is not None, 'Metadata path is required'
        assert labels_map is not None, 'Labels map is required'
        assert audio_config is not None, 'Audio config is required.'

        # Read manifest, metadata, labels map
        self.get_encoded_metadata(metadata_path)
        with open(labels_map) as f:
            self.labels_map = json.load(f)
        manifest = pd.read_csv(manifest_path)
        self.files = manifest["files"].values
        self.labels = manifest["labels"].values
        assert len(self.files) == len(self.labels), 'File-label length mismatch.'

        self.get_audio_config(audio_config)
        self.get_spec_parser()
        self.augment = augment
        if self.augment:
            self.mask_param = mask_param
            self.get_augments()

        self.get_weighted_sampler()

    def get_encoded_metadata(self, metadata_path):
        metadata = pd.read_csv(metadata_path)
        for column in ['gender', 'respiratory_condition', 'fever_muscle_pain']:
            metadata[column] = metadata[column].astype('category').cat.codes
        self.metadata = metadata

    def get_weighted_sampler(self):
        from torch.utils.data import WeightedRandomSampler
        # Initialize an empty array for weights
        weights = []
        
        # Calculate label counts
        label_counts = {}
        for labels in self.labels:
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
                
        for labels in self.labels:
            weight = sum(1.0 / label_counts[label] for label in labels)
            weights.append(weight)
        
        weights = torch.FloatTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights))
        self.weighted_sampler = sampler

    def get_augments(self):
        self.mask_param = self.mask_param
        random.seed(42)
        torch.manual_seed(42)
        self.time_mask = TimeMasking(time_mask_param=self.mask_param, iid_masks=True)
        self.freq_mask = FrequencyMasking(freq_mask_param=self.mask_param, iid_masks=True)

    def get_spec_parser(self):
        self.spec_parser = AudioParser(
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop_len,
            feature=self.feature,
        )
    
    def get_audio_config(self, audio_config):
        self.sr = audio_config.get("sample_rate", "24000")
        self.n_fft = audio_config.get("n_fft", 511)
        self.win_len = audio_config.get("win_len", self.n_fft)
        self.hop_len = audio_config.get("hop_len", self.n_fft//2)
        self.normalize = audio_config.get("normalize", True)
        self.min_duration = audio_config.get("min_duration", None)
        self.feature = audio_config.get("feature", "spectrogram")

    def transform(self, spec):
        spec = torch.from_numpy(spec)
        spec = spec.expand(1, -1, -1)
        return spec
    
    def masking(self, spec):
        for mask in range(2):
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)
        return spec
    
    def __get_audio__(self, f):
        audio = load_audio(f, self.sr, self.min_duration)
        return audio

    def __get_feature__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real, comp = self.spec_parser(audio)
        return real, comp
    
    def __parse_labels__(self, lbl: str) -> torch.Tensor:
        label_tensor = torch.zeros(len(self.labels_map)).float()
        label_tensor[self.labels_map[lbl]] = 1
        return label_tensor

    def __get_feats__(self, f):
        chunk_id = os.path.basename(f).replace('wav', '')
        mask = [uuid in chunk_id for uuid in self.metadata.uuid]
        feats = self.metadata[mask][['SNR', 
                                     'age', 
                                     'gender', 
                                     'respiratory_condition', 
                                     'fever_muscle_pain']].values.flatten().tolist()
        return torch.tensor(feats)

    def __get_item_helper__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        f = self.files[index]
        preprocessed_audio = self.__get_audio__(f)
        real, comp = self.__get_feature__(preprocessed_audio)
    
        lbl = self.labels[index]
        label_tensor = self.__parse_labels__(lbl)

        feats = self.__get_feats__(f)
        return real, comp, feats, label_tensor
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real, comp, feats, targets = self.__get_item_helper__(index)
        real = self.transform(real)
        if self.augment:
            real = self.masking(real)

        return real, feats, targets
    
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    from utils.config_parser import parse_config
    from torch.utils.data import DataLoader
    config = parse_config('configs/ssformer.cfg')
    manifest_path = r"C:\Users\vismi\Documents\datasets\covid19-cough\manidests\no_labeled_chunk.csv"

    dataset = SpectrogramDataset(manifest_path = manifest_path,
                                 labels_map= None,
                                audio_config = config['audio_config'],
                                mode = "multiclass")
    
    dataloader = DataLoader(dataset, batch_size=1)
    print(next(iter(dataloader[0].shape)))