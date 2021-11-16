import torch
from torch.utils.data import DataLoader

from IPython.display import clear_output

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
plt.style.use("ggplot")

from configs import *

from kws.datasets import SpeechCommandDataset
from kws.augmentations import WaveAugs, LogMelSpec
from kws.collate_fn import Collator

from kws.sampler.sampler import get_sampler
from kws.utils.seed_everything import seed_everything

from kws.models import *
from kws.trainer import *


def main(config):
    seed_everything(seed=config.seed)

    if config.verbose:
        print(f"The training proccess will be performed on {config.device}.")

    # Load the data
    dataset = SpeechCommandDataset(path_to_dir=config.path_to_data, keywords=config.keyword)

    indexes = torch.randperm(len(dataset))
    train_indexes = indexes[:int(len(dataset) * config.train_ratio)]
    val_indexes = indexes[int(len(dataset) * config.train_ratio):]

    df_train = dataset.csv.iloc[train_indexes].reset_index(drop=True)
    df_val = dataset.csv.iloc[val_indexes].reset_index(drop=True)

    train_set = SpeechCommandDataset(csv=df_train, transform=WaveAugs(config))
    val_set = SpeechCommandDataset(csv=df_val)

    train_sampler = get_sampler(train_set.csv["label"].values)
    val_sampler = get_sampler(val_set.csv["label"].values)

    train_loader = DataLoader(train_set, 
                              batch_size=config.batch_size,
                              shuffle=False,   # because of our sampler with randomness inside
                              collate_fn=Collator(),
                              sampler=train_sampler,
                              num_workers=config.num_workers, 
                              pin_memory=True)

    val_loader = DataLoader(val_set, 
                            batch_size=config.batch_size,
                            shuffle=False, 
                            collate_fn=Collator(),
                            sampler=val_sampler,
                            num_workers=config.num_workers, 
                            pin_memory=True)

    if config.verbose:
        print("The data is loaded, and the minor target class is oversampled.")

    melspec_train = LogMelSpec(is_train=True, config=config)
    melspec_val = LogMelSpec(is_train=False, config=config)

    if config.model_type == "base":
        model = CRNNBase(config).to(config.device)
    else:
        raise ValueError("Error in model type definition.")

    history = defaultdict(list)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    for i in range(config.num_epochs):
        train_epoch(model, optimizer, train_loader, melspec_train, config.device)
        auc_fa_fr = validation(model, val_loader, melspec_val, config.device)
        history["val_auc_fa_fr"].append(auc_fa_fr)

        clear_output()

        plt.plot(history["val_auc_fa_fr"])
        plt.ylabel("Metric")
        plt.xlabel("Epoch")
        plt.show()

        if config.verbose:
            print(f"Epoch {i}: AUC_FA_FR = {auc_fa_fr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")

    parser.add_argument("-m",
                        "--model",
                        metavar="model",
                        default=None,
                        required=True,
                        type=str,
                        help="model type")

    args = parser.parse_args()

    if args.model == "base":
        config = ConfigBase()
    else:
        raise NotImplementedError()

    main(config)
