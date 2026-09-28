from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_features, read_manifest, split_records
from .metrics import classification_metrics
from .model import GLADMIL
from .utils import load_config, resolve_device, seed_everything, write_json


def evaluate(model, rows, device, groups, gamma=0.5):
    model.eval(); labels, probs, loss_values = [], [], []
    with torch.no_grad():
        for row in rows:
            output = model(load_features(row, device), groups, np.random.default_rng(104729), random_partition=False)
            target = torch.tensor([row.label], device=device)
            full = output["full_logits"]; tier2 = output["tier2_logits"]
            p = (1.0 - gamma) * torch.softmax(full, 1)[0, 1] + gamma * torch.softmax(tier2, 1)[0, 1]
            labels.append(row.label); probs.append(float(p)); loss_values.append(float(F.cross_entropy(tier2, target)))
    metrics = classification_metrics(labels, probs); metrics["loss"] = float(np.mean(loss_values)); return metrics


def select_gamma(model, rows, device, groups, step):
    candidates = np.arange(0.0, 1.0 + step / 2, step)
    scored = [(evaluate(model, rows, device, groups, float(gamma))["auc"], float(gamma)) for gamma in candidates]
    best_auc = max(-np.inf if np.isnan(auc) else auc for auc, _ in scored)
    tied = [gamma for auc, gamma in scored if (not np.isnan(auc) and abs(auc - best_auc) <= 1e-12)]
    return min(tied, key=lambda gamma: (abs(gamma - 0.5), gamma))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config_path = Path(args.config).resolve(); config = load_config(config_path); seed_everything(int(config["seed"])); device = resolve_device(config.get("device", "auto"))
    for key in ("manifest", "output_dir"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else config_path.parent / value)
    splits = split_records(read_manifest(config["manifest"])); output_dir = Path(config["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    model = GLADMIL(config["input_dim"], config.get("num_classes", 2), config.get("window_size", 64), config.get("adapter_dropout", .25), config.get("adapter_max_scale", .05)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config["epochs"])))
    best_auc, best_loss, stale, best_path = -np.inf, np.inf, 0, output_dir / "best.pt"
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train(); losses = []; order = np.random.default_rng(int(config["seed"]) + epoch).permutation(len(splits["train"]))
        for index in order:
            row = splits["train"][int(index)]; output = model(load_features(row, device), config["num_groups"], np.random.default_rng(int(config["seed"]) * 1000003 + epoch * 1009 + int(index)), random_partition=True)
            target = torch.tensor([row.label], device=device)
            loss = F.cross_entropy(output["full_logits"], target) + config["alpha_pseudobag"] * F.cross_entropy(output["sub_logits"], target.repeat(output["sub_logits"].shape[0])) + config["beta_tier2"] * F.cross_entropy(output["tier2_logits"], target)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 5.0)); optimizer.step(); losses.append(float(loss.detach()))
        gamma = select_gamma(model, splits["val"], device, config["num_groups"], config.get("gamma_grid_step", .05)); val = evaluate(model, splits["val"], device, config["num_groups"], gamma); val["epoch"] = epoch; val["gamma"] = gamma; val["train_loss"] = float(np.mean(losses)); history.append(val)
        print(f"epoch={epoch:03d} train_loss={val['train_loss']:.4f} val_auc={val['auc']:.4f} gamma={gamma:.2f}", flush=True)
        improved = val["auc"] > best_auc or (val["auc"] == best_auc and val["loss"] < best_loss)
        if improved:
            best_auc, best_loss, stale = val["auc"], val["loss"], 0; torch.save({"model": model.state_dict(), "config": config, "gamma": gamma, "epoch": epoch}, best_path)
        else: stale += 1
        scheduler.step()
        if stale >= int(config["patience"]) and epoch > int(config.get("early_stop_after", 0)): break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"]); gamma = float(checkpoint["gamma"])
    result = {"best_epoch": checkpoint["epoch"], "gamma": gamma, "validation": evaluate(model, splits["val"], device, config["num_groups"], gamma), "test": evaluate(model, splits["test"], device, config["num_groups"], gamma) if splits["test"] else None, "history": history}
    write_json(output_dir / "metrics.json", result); print(result)


if __name__ == "__main__": main()
