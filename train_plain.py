"""Plain-PyTorch training loop: same data, model and loss as train.py, no Lightning.

Mirrors what spt.Module + pl.Trainer do for this config:
  bf16 autocast -> loss -> backward -> clip grad norm 1.0 -> AdamW step
  -> per-step LR schedule (1% linear warmup, cosine to 0) -> zero_grad.

Skipped vs train.py: multi-GPU, wandb, resumable checkpoints, deterministic algorithms.

    python train_plain.py
"""
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from module import SIGReg
from train import lejepa_forward
from train_no_hydra import build_loaders, build_model, load_cfg


def to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def main():
    cfg = load_cfg("pusht")
    run_dir = Path("outputs") / f"plain-{cfg.subdir}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. data (must run before build_model: fills cfg.wm.action_dim etc.)
    train, val = build_loaders(cfg)

    # 2. model + loss regularizer (SIGReg has buffers only, nothing to train)
    model = build_model(cfg).to(device)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)

    # 3. optimizer + LR schedule
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    total_steps = cfg.trainer.max_epochs * len(train)
    warmup_steps = max(1, int(0.01 * total_steps))

    def lr_factor(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    # lejepa_forward expects `self` with .model, .sigreg, .log_dict
    logs = {}
    shim = SimpleNamespace(
        model=model, sigreg=sigreg, log_dict=lambda d, **_: logs.update(d)
    )

    # 4. loop
    for epoch in range(cfg.trainer.max_epochs):
        model.train()
        t0 = time.time()
        for step, batch in enumerate(train):
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = lejepa_forward(shim, batch, "fit", cfg)

            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clip_val)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(train)} "
                    f"loss {logs['fit/loss']:.4f} pred {logs['fit/pred_loss']:.4f} "
                    f"sigreg {logs['fit/sigreg_loss']:.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s"
                )

        model.eval()
        val_losses = []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            for batch in val:
                out = lejepa_forward(shim, to_device(batch, device), "validate", cfg)
                val_losses.append(out["loss"].item())
        print(f"epoch {epoch} val loss {sum(val_losses) / len(val_losses):.4f}")

        torch.save(model, run_dir / f"{cfg.output_model_name}_epoch_{epoch + 1}_object.ckpt")


if __name__ == "__main__":
    main()
