import torch
from torch.utils.data import DataLoader

from IPython.display import clear_output

#import hydra
#from omegaconf import DictConfig

from collections import defaultdict
from thop import profile

import matplotlib.pyplot as plt
plt.style.use("ggplot")

from configs import *

from kws.datasets import SpeechCommandDataset
from kws.augmentations import WaveAugs, LogMelSpec
from kws.collate_fn import Collator

from kws.sampler import *
from kws.models import *
from kws.trainer import *
from kws.utils import *

#@hydra.main(config_path="../config", config_name="config")
def main(config):#: DictConfig):
    seed_everything(seed=config.seed)

    if config.verbose:
        print(f"The training process will be performed on {config.device}.")

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
                              shuffle=False,   # because our sampler with randomness inside
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

    if config.use_distillation:
        base_model = None
        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        additional_model = CRNNStreaming(config).to(config.device)
        additional_model.load_state_dict(torch.load(config.path_to_load)["state_dict"])
        
    else:
        base_model = CRNNStreaming(config).to(config.device)
        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    if config.verbose:
        print("The training process is started.")

    history = defaultdict(list)
    with Timer(verbose=config.verbose):
        for epoch in range(config.num_epochs):
            if config.use_distillation:
                distill_train_epoch(teacher_model=additional_model, student_model=base_model,
                                    optimizer=optimizer, loader=train_loader, 
                                    log_melspec=melspec_train, device=config.device,
                                    temperature=config.temperature, alpha = config.alpha)
            else:
                train_epoch(model=base_model, optimizer=optimizer, 
                            loader=train_loader, log_melspec=melspec_train, device=config.device)

            auc_fa_fr = validation(base_model, val_loader, melspec_val, config.device)
            history["val_auc_fa_fr"].append(auc_fa_fr)

            if auc_fa_fr <= min(history["val_auc_fa_fr"]):
                arch = type(base_model).__name__
                state = {
                    "arch": arch,
                    "epoch": epoch,
                    "state_dict": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config
                }
                best_path = config.path_to_save + f"{config.model_name}_best.pth"
                torch.save(state, best_path)

            clear_output()

            plt.plot(range(1, epoch + 2), history["val_auc_fa_fr"])
            plt.title("Validation AUC of FA-FR Curve")
            plt.ylabel("Metric")
            plt.xlabel("Epoch")

            plt.show()

            if config.verbose:
                print(f"Epoch {epoch + 1}: AUC_FA_FR = {auc_fa_fr:.6}")
    
    if config.verbose:
        print(f"Number of parameters: {get_num_params(base_model)}.")
        print(f"Size in megabytes: {get_size_in_megabytes(base_model):.4}.")


#if __name__ == "__main__":
#    main()
