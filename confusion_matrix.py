"""Script for creating confusion matrix, precision, recall, ROC"""

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch import nn
from tqdm import tqdm

import wandb

# from utils.trainer import train, evaluate
from src.data.dataset import SpectrogramDataset
from src.models.KWT import KWT
from utils.config_parser import parse_config


def test_pipeline(config):

    # Setting up device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Getting test data set
    test_set = SpectrogramDataset(
        config["eval_manifest_path"], config["labels_map"], config["audio_config"]
    )
    classes = {"COVID-19": 0, "healthy": 1, "symptomatic": 2}  # The 3 classes

    # For debugging/testing on smaller test data set

    # test_set.files = test_set.files[:100]
    # test_set.len = len(test_set.files)
    # test_set.labels = test_set.labels[:100]

    # checkpoint = torch.load("outputs/augmented_supervised_best.pth") # loading checkpoint
    checkpoint = torch.load(args.ckpt,map_location=device)
    model = KWT(**config["hparams"]["KWT"])  # loading model
    model.to(device)
    # Sending model to device
    model.load_state_dict(checkpoint["model_state_dict"])  # loading state_dict
    model.eval()  # setting model to evaluation mode

    # Loading test data and creating data loader
    test_loader = DataLoader(
        test_set, batch_size=1
    )  # batch size 1, append later for memory preservation
    # print(len(test_loader))

    # Evaluation loop
    print("Initiating testing...")
    predicted_labels = []
    true_labels = []

    # iterate over test data
    for spectrogram, labels in tqdm(test_loader):
        spectrogram = spectrogram.to(device)  # sending spectograms to device
        labels = labels.to(device)  # sending labels to device
        output = model(spectrogram)  # feeding data to network
        predicted_labels.append(
            output.argmax(1)
        )  # argmax to find "hot one" in one hot encoding
        true_labels.append(
            labels.argmax(1)
        )  # argmax to find "hot one" in one hot encoding

    predicted_labels = torch.flatten(
        torch.tensor(predicted_labels)
    )  # flattening dimensions
    true_labels = torch.flatten(torch.tensor(true_labels))  # flattening dimensions
    # print(pd.Series(predicted_labels).value_counts())
    # print(pd.Series(true_labels).value_counts())

    # Creating confusion matrix using sklearn function
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plotting confusion matrix
    df_cm = pd.DataFrame(
        cm / np.sum(cm, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    confusion_matrix_fig = plt.figure(figsize=(12, 7))
    plt.title("Confusion Matrix")
    sns.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # The ROC curve is a plot of the true positive rate (TP) against the false positive rate (FP) at various threshold settings.
    # Each point on the ROC curve represents a sensitivity/specificity pair corresponding to a particular decision threshold.
    # The AUC measures the entire two-dimensional area underneath the ROC curve from (0,0) to (1,1).

    # Interpreting the AUC:
    # AUC = 1: Perfect classifier. It means the model has perfect discrimination ability, achieving a TPR of 1 (sensitivity) while maintaining an FPR of 0 (specificity) across all thresholds.
    # AUC = 0.5: Random classifier. It indicates that the model has no discrimination ability better than random guessing. The ROC curve coincides with the diagonal line (the line of no-discrimination).
    # AUC < 0.5: Worse than random classifier. It suggests that the model's predictions are inversely related to the true labels, indicating a model with poor performance.

    # logging to    #logging confusion matrix, precision-recall curve, ROC-curve to wandb
    wandb.log({"confusion_matrix.png": wandb.Image(confusion_matrix_fig)})
    # Logging precision-recall curve figure
    wandb.log({"precision_recall_curve.png": wandb.Image(precision_recall_fig)})
    # Logging ROC curve figure
    wandb.log({"ROC_curve.png": wandb.Image(ROC_curve_fig)})


def main(args):
    """
    Loads best checkpoint from training run
    Runs test loop using the test data
    Plots confusion matrix, precision-recall curve and ROC-curve over results

    """

    config = parse_config(args.conf)

    """"for logging in wandb"""

    # below code is for logging everything in wandb
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"]

    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()

        elif os.environ.get("WANDB_API_KEY", False):
            print("Found API key from env variable.")

        else:
            wandb.login()

        with wandb.init(
            project=config["exp"]["proj_name"],
            name=config["exp"]["exp_name"],
            config=config["hparams"],
            entity=config["exp"]["entity"],
        ):
            test_pipeline(config)
    else:

        test_pipeline(config)

    # Precision and Recall
    precision_per_class = []
    recall_per_class = []

    # finding precision adn recall for each class with sklearn
    for i in range(len(classes)):
        precision = precision_score(true_labels == i, predicted_labels == i)
        recall = recall_score(true_labels == i, predicted_labels == i)
        precision_per_class.append(precision)
        recall_per_class.append(recall)

    print("Precision of class 'Covid': ", precision_per_class[0])
    print("Precision of class 'healthy': ", precision_per_class[1])
    print("Precision of class 'symptomatic': ", precision_per_class[2])

    print("Recall of class 'Covid': ", recall_per_class[0])
    print("Recall of class 'healthy': ", recall_per_class[1])
    print("Recall of class 'symptomatic': ", recall_per_class[2])

    # Plotting precision-recall curve

    # The precision-recall curve shows the tradeoff between precision and recall for different threshold.
    # A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate,
    # and high recall relates to a low false negative rate. (from sklearn documentation) --> so we want both high recall and high precision for a good model

    precision_recall_fig = plt.figure(figsize=(12, 7))

    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(
            true_labels == i, predicted_labels == i
        )
        class_name = list(classes.keys())[list(classes.values()).index(i)]
        plt.plot(
            recall,
            precision,
            label=" {} (Precision = {:.2f}, Recall = {:.2f})".format(
                class_name, precision_per_class[i], recall_per_class[i]
            ),
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("precision_recall_curve.png")
    plt.show()

    # Plotting ROC curve

    # ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis.
    # This means that the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one.
    # This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.
    # The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate while minimizing the false positive rate.  (from sklearn documentation)

    ROC_curve_fig = plt.figure(figsize=(12, 7))
    # for multi-class classification, it is necessary to binarize the output. One ROC curve can be drawn per label.
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(true_labels == i, predicted_labels == i)
        roc_auc = auc(fpr, tpr)
        class_name = list(classes.keys())[list(classes.values()).index(i)]
        plt.plot(
            fpr,
            tpr,
            lw=2,
            label="ROC curve for {} (AUC = {:.2f})".format(class_name, roc_auc),
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # ROC_fig = plt.figure(figsize = (12,7))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive (FP) Rate")
    plt.ylabel("True Positive (TP) Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("ROC_curve.png")
    plt.show()

    # The ROC curve is a plot of the true positive rate (TP) against the false positive rate (FP) at various threshold settings.
    # Each point on the ROC curve represents a sensitivity/specificity pair corresponding to a particular decision threshold.
    # The AUC measures the entire two-dimensional area underneath the ROC curve from (0,0) to (1,1).

    # Interpreting the AUC:
    # AUC = 1: Perfect classifier. It means the model has perfect discrimination ability, achieving a TPR of 1 (sensitivity) while maintaining an FPR of 0 (specificity) across all thresholds.
    # AUC = 0.5: Random classifier. It indicates that the model has no discrimination ability better than random guessing. The ROC curve coincides with the diagonal line (the line of no-discrimination).
    # AUC < 0.5: Worse than random classifier. It suggests that the model's predictions are inversely related to the true labels, indicating a model with poor performance.

    # logging to    #logging confusion matrix, precision-recall curve, ROC-curve to wandb
    wandb.log({"confusion_matrix.png": wandb.Image(confusion_matrix_fig)})
    # Logging precision-recall curve figure
    wandb.log({"precision_recall_curve.png": wandb.Image(precision_recall_fig)})
    # Logging ROC curve figure
    wandb.log({"ROC_curve.png": wandb.Image(ROC_curve_fig)})


def main(args):
    """
    Loads best checkpoint from training run
    Runs test loop using the test data
    Plots confusion matrix, precision-recall curve and ROC-curve over results

    """

    config = parse_config(args.conf)

    """"for logging in wandb"""

    # below code is for logging everything in wandb
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"]

    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()

        elif os.environ.get("WANDB_API_KEY", False):
            print("Found API key from env variable.")

        else:
            wandb.login()

        with wandb.init(
            project=config["exp"]["proj_name"],
            name=config["exp"]["exp_name"],
            config=config["hparams"],
            entity=config["exp"]["entity"],
        ):
            test_pipeline(config)
    else:

        test_pipeline(config)


if __name__ == "__main__":

    parser = ArgumentParser("Driver code.")
    parser.add_argument(
        "--conf", type=str, required=True, help="Path to config.yaml file."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint file.", default=None
    )
    parser.add_argument(
        "--id",
        type=str,
        required=False,
        help="Optional experiment identifier.",
        default=None,
    )
    args = parser.parse_args()

    main(args)
