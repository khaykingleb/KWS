from tqdm import tqdm

import torch
import torch.nn.functional as F

from kws.metrics import count_FA_FR, get_auc_FA_FR


def distill_train_epoch(teacher_model, teacher_optimizer, student_model, student_optimizer,
                        loader, log_melspec, device):
    teacher_model.train()
    student_model.train()

    for _, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        teacher_optimizer.zero_grad()


def train_epoch(model, optimizer, loader, log_melspec, device):
    model.train()
    for _, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        optimizer.zero_grad()

        # Run model # with autocast():
        logits = model(batch)
        
        # We need probabilities so we use softmax & CE separately
        probs = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        # Logging
        argmax_probs = torch.argmax(probs, dim=-1)
        FA, FR = count_FA_FR(argmax_probs, labels)
        accuracy = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

    return accuracy


@torch.no_grad()
def validation(model, loader, log_melspec, device):
    model.eval()

    val_losses, FAs, FRs = [], [], []
    all_probs, all_labels = [], []
    for _, (batch, labels) in tqdm(enumerate(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        output = model(batch)
        # We need probabilities so we use softmax & CE separately
        probs = F.softmax(output, dim=-1)
        loss = F.cross_entropy(output, labels)

        # Logging
        argmax_probs = torch.argmax(probs, dim=-1)
        all_probs.append(probs[:, 1].cpu())
        all_labels.append(labels.cpu())
        val_losses.append(loss.item())
        FA, FR = count_FA_FR(argmax_probs, labels)
        FAs.append(FA)
        FRs.append(FR)

    # Area under FA/FR curve for whole loader
    auc_fa_fr = get_auc_FA_FR(torch.cat(all_probs, dim=0).cpu(), all_labels)
    return auc_fa_fr, val_losses, FAs, FRs
