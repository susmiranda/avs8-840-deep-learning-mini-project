import os
import math
import time

import wandb
import tqdm

import torch
from torch.utils.data import DataLoader
from torch import optim, nn

from src.models.ssformer import SSTransformer
from src.utils.misc import save_model


def compute_var(y: torch.Tensor):
    """
    Function for computing standard deviation of target
    :param y:
    :return: standard deviation of y
    """
    y = y.view(-1, y.size(-1))
    return torch.sqrt(y.var(dim=0) + 1e-6).mean()

def train_single_batch(net: SSTransformer, data: torch.Tensor, mask: torch.Tensor, optimizer: optim.Optimizer,
                       criterion, device: torch.device):
    """
    Performs single training step for Data2Vec model
    :param net: Data2Vec model
    :param data: input data
    :param mask: mask for student input
    :param optimizer: torch optimizer
    :param criterion: torch criterion
    :param device: device for model and optimizer
    :return: loss, target variance and prediction variance
    """

    # Send data to GPU if available
    data = data.to(device)

    # Reset gradients
    optimizer.zero_grad()

    # Put the same input for student and teacher. Get outputs (final hidden states)
    predictions, targets = net(data, data, mask)
    scale = math.sqrt(predictions.size(dim=-1))
    loss = criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        target_var = compute_var(targets.float())
        prediction_var = compute_var(predictions.float())
    return loss.item(), target_var.item(), prediction_var.item()

@torch.no_grad()
def evaluate(net: SSTransformer, mask_generator, criterion, dataloader: DataLoader,
             device: torch.device):
    """
    Evaluates Data2Vec model
    :param net: Data2Vec model
    :param mask_generator: mask generator for student input mask
    :param criterion: torch criterion
    :param dataloader: torch dataloader
    :param device: device for model and optimizer
    :return: loss, target variance and prediction variance
    """

    net.eval()  # Set model in evaluation mode
    # Initialise metrics
    running_loss = 0.0
    running_target_var = 0.0
    running_prediction_var = 0.0

    for data in tqdm(dataloader):
        data = data.to(device)
        batch_size = data.size(dim=0)
        audio_length = data.size(dim=-1)
        mask = mask_generator(shape=(batch_size, audio_length)).to(device)
        mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device), mask], dim=1).bool()

        predictions, targets = net(data, data, mask)
        scale = math.sqrt(predictions.size(dim=-1))
        loss = criterion(predictions.float(), targets.float()).sum(dim=-1).sum().div(scale)
        target_var = compute_var(targets.float())
        prediction_var = compute_var(predictions.float())
        running_loss += loss.item()
        running_target_var += target_var.item()
        running_prediction_var += prediction_var.item()

    avg_loss = running_loss / len(dataloader.dataset)
    avg_target_var = running_target_var / len(dataloader)
    avg_prediction_var = running_prediction_var / len(dataloader)
    return avg_loss, avg_target_var, avg_prediction_var


def train(net: nn.Module, mask_generator, optimizer: optim.Optimizer, criterion, train_loader: DataLoader,
          validation_loader: DataLoader, schedulers: dict, config: dict):
    """
    Trains Data2Vec model
    :param net: Data2Vec model
    :param mask_generator:
    :param optimizer: torch optimizer
    :param criterion: torch criterion
    :param train_loader: training set loader
    :param validation_loader: validation set loader
    :param schedulers: learning rate scheduler
    :param config: model and training config
    """

    step = 0
    best_avg_loss = 0.0
    n_batches = len(train_loader)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

    ############################
    # start training
    ############################
    net.train()

    for epoch in range(config["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss = 0.0
        running_target_var = 0.
        running_prediction_var = 0.
        for batch_index, data, in enumerate(train_loader):
            batch_size = data.size(dim=0)
            audio_length = data.size(dim=-1)
            if schedulers["warmup"] is not None and epoch < config["hparams"]["scheduler"]["n_warmup"]:
                schedulers["warmup"].step()

            elif schedulers["scheduler"] is not None:
                schedulers["scheduler"].step()

            ####################
            # optimization step
            ####################
            mask = mask_generator(shape=(batch_size, audio_length)).to(device)
            mask = torch.cat([torch.zeros(batch_size, 1, device=mask.device), mask], dim=1).bool()

            loss, target_var, prediction_var = train_single_batch(net, data, mask, optimizer, criterion, device)
            net.ema_step()
            running_loss += loss
            running_target_var += target_var
            running_prediction_var += prediction_var

            if not step % config["exp"]["log_freq"]:
                log_dict = {"epoch": epoch, "loss": loss, "lr": optimizer.param_groups[0]["lr"],
                            "target_var": target_var, "prediction_var": prediction_var}
                wandb.log(log_dict, step=step)

            step += 1

        #######################
        # epoch complete
        #######################

        log_dict = {"epoch": epoch, "time_per_epoch": time.time() - t0,
                    "avg_train_target_var": running_target_var / n_batches,
                    "avg_train_prediction_var": running_prediction_var / n_batches,
                    "avg_loss_per_ep": running_loss / len(train_loader.dataset)}
        wandb.log(log_dict, step=step)

        if not epoch % config["exp"]["val_freq"]:
            avg_val_loss, avg_val_target_var, avg_val_prediction_var = evaluate(net, mask_generator, criterion,
                                                                                validation_loader, device)
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss,
                        "avg_val_target_var": avg_val_target_var, "avg_val_prediction_var": avg_val_prediction_var}
            wandb.log(log_dict, step=step)

            # save best validation checkpoint
            if avg_val_loss < best_avg_loss or epoch == config["exp"]["val_freq"]:
                best_avg_loss = avg_val_loss
                save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
                save_model(epoch, avg_val_loss, save_path, net, optimizer, log_file)
                save_path = os.path.join(config["exp"]["save_dir"], "best_encoder.pth")
                save_model(epoch, avg_val_loss, save_path, net.encoder, optimizer, log_file)

    ###########################
    # training complete
    ###########################
                
    # Save the best model on wandb
    wandb.save(save_path)

    avg_val_loss, avg_val_target_var, avg_val_prediction_var = evaluate(net, mask_generator, criterion, validation_loader,
                                                                        device)
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss,
                "avg_val_target_var": avg_val_target_var, "avg_val_prediction_var": avg_val_prediction_var}

    wandb.log(log_dict, step=step)

    # save final checkpoint
    save_path = os.path.join(config["exp"]["save_dir"], "last.pth")
    save_model(epoch, avg_val_loss, save_path, net, optimizer, log_file)
    save_path = os.path.join(config["exp"]["save_dir"], "last_encoder.pth")
    save_model(epoch, avg_val_loss, save_path, net.encoder, optimizer, log_file)
    wandb.save(save_path)