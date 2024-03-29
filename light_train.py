
import wandb

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from utils.config_parser import parse_config
from utils.light_modules import LightningKWT
from src.models.KWT import KWT
from src.data.dataset import SpectrogramDataset


def training_pipeline(config, logger, ckpt_path):

    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Initialize ViT
    vit = KWT(**config['hparams']['KWT'])
    vit.to(device);

    # Initialize lightning module from checkpoint if provided
    if ckpt_path:
        model = LightningKWT.load_from_checkpoint(ckpt_path, model=vit, config=config)
    else:
        model = LightningKWT(model=vit, config=config)
    model.to(device)

    # Make datasets
    train_set = SpectrogramDataset(config['tr_manifest_path'], labels_map=config['labels_map'], audio_config=config['audio_config'], augment=True)
    val_set = SpectrogramDataset(config['val_manifest_path'], labels_map=config['labels_map'], audio_config=config['audio_config'])

    if config['dev_mode']:
        train_set.files = train_set.files[:50]
        train_set.len = len(train_set.files)
        val_set.files = val_set.files[:50]
        val_set.len = len(val_set.files)
        config['hparams']['batch_size'] = 25

    # Make dataloaders
    train_loader = DataLoader(train_set, batch_size=config['hparams']['batch_size'], num_workers=5, sampler=train_set.weighted_sampler)
    val_loader = DataLoader(val_set, batch_size=config['hparams']['batch_size'], num_workers=5, sampler=val_set.weighted_sampler)

    # Create Callbacks
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", verbose=True)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=config['hparams']['early_stopping_patience'], verbose=True)
    callbacks = [model_checkpoint, early_stopping]

    trainer = L.Trainer(max_epochs=config['hparams']['n_epochs'], 
                        logger=logger,
                        callbacks=callbacks,
                        log_every_n_steps=50,
                        strategy='ddp_find_unused_parameters_true')
    trainer.fit(model, train_loader, val_loader)

def main(args):

    config = parse_config(args.config)
    if args.tr_manifest_path:
        config['tr_manifest_path'] = args.tr_manifest_path
    if args.val_manifest_path:
        config['val_manifest_path'] = args.val_manifest_path
    if args.labels_map:
        config['labels_map'] = args.labels_map
    config['dev_mode'] = args.dev_mode
    
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + args.id
    
    if config["exp"]["wandb"]:
        wandb.login()
        logger = WandbLogger(project=config["exp"]["proj_name"],
                                name=config["exp"]["exp_name"],
                                entity=config["exp"]["entity"],
                                config=config["hparams"],
                                log_model=True,
                                save_dir='outputs/')
        training_pipeline(config, logger, args.ckpt_path)

    else:
        logger = None
        training_pipeline(config, logger, args.ckpt_path)


if __name__ == '__main__':
    from argparse import ArgumentParser

    ap = ArgumentParser("Driver code")
    ap.add_argument('config', type=str, help='Path to configuration file')
    ap.add_argument('--id', type=str, help='Unique experiment identifier')
    ap.add_argument('--tr_manifest_path', type=str, help='Path to the unlabeled train data manifest.')
    ap.add_argument('--val_manifest_path', type=str, help='Path to the unlabeled val data manifest.')
    ap.add_argument('--labels_map', type=str, help='Path to lbl_map.json')
    ap.add_argument('--ckpt_path', type=str, help='Path to model checkpoint.')
    ap.add_argument('--dev_mode', action='store_true', help='Flag to limit the dataset for testing purposes.')
    args = ap.parse_args()

    main(args)