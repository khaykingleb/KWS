import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader

from IPython.display import clear_output

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

THRESHOLD = 5e-5 * 1.1


def main(config, small_config=None):
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

    train_set = SpeechCommandDataset(csv=df_train, transform=WaveAugs())
    val_set = SpeechCommandDataset(csv=df_val)

    train_sampler = get_sampler(train_set.csv["label"].values)

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
                            num_workers=config.num_workers, 
                            pin_memory=True)

    if config.verbose:
        print("The data is loaded, and the minor target class is oversampled.")

    melspec_train = LogMelSpec(is_train=True, config=config)
    melspec_val = LogMelSpec(is_train=False, config=config)

    if small_config is not None:
        if small_config.use_distillation:
            base_model = CRNNStreaming(small_config)
            optimizer = torch.optim.Adam(
                base_model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )

            additional_model = CRNNStreaming(config).to(config.device)
            additional_model.load_state_dict(torch.load(small_config.path_to_load)["state_dict"])
        
    else:
        base_model = CRNNStreaming(config).to(config.device)
        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    if small_config is not None:
        if (config.use_quantization and not small_config.use_distillation) or \
        (not config.use_quantization and small_config.use_quantization):
            assert config.device is "cpu"
            base_model = torch.quantization.quantize_dynamic(
                base_model, {nn.GRU, nn.Linear}, dtype=config.quantization_type
            )

        elif config.use_quantization and small_config.use_quantization:
            assert config.device is "cpu" and small_config.device is "cpu"

            base_model = torch.quantization.quantize_dynamic(
                base_model, {nn.GRU, nn.Linear}, dtype=small_config.quantization_type
            )
            additional_model = torch.quantization.quantize_dynamic(
                additional_model, {nn.GRU, nn.Linear}, dtype=config.quantization_type
            )
    
    elif config.use_quantization:
        assert config.device is "cpu"
        base_model = torch.quantization.quantize_dynamic(
            base_model, {nn.GRU, nn.Linear}, dtype=config.quantization_type
        )

    if config.verbose:
        print("The training process is started.")

    history = defaultdict(list)
    with Timer(verbose=config.verbose) as timer:
        for epoch in range(config.num_epochs):
            if small_config is not None:
                if small_config.use_distillation:
                    distill_train_epoch(teacher_model=additional_model, student_model=base_model,
                                        optimizer=optimizer, loader=train_loader, 
                                        log_melspec=melspec_train, device=config.device,
                                        temperature=small_config.temperature, alpha = small_config.alpha)

            elif small_config is None or (small_config is not None and not small_config.use_distillation):
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
            
            if auc_fa_fr <= THRESHOLD:
                if config.model_name == "base_2x64" and auc_fa_fr <= THRESHOLD / 1.1:
                    print("Achieved the threshold successively.")
                    break
                break

        time = timer.get_time()
    
    macs, num_params = profile(base_model, torch.zeros(1, 1, 40, 50).to(config.device), verbose=False)
    size = get_size_in_megabytes(base_model)

    result = {
        "model": config.model_name,
        "macs": macs, 
        "num_params": num_params,
        "time": time
    }

    if config.verbose:
        print(f"MACs: {macs}.")
        print(f"Parameters: {num_params}.")
        print(f"Size in megabytes: {size:.4}.") 
    
    return result
