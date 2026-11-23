"""Plain-PyTorch training loop with the JEPA forward pass written out explicitly.

Same as train_plain.py, but instead of importing `lejepa_forward` from train.py
(which expects a Lightning-style `self`), the forward pass and losses are spelled
out in `jepa_losses(model, sigreg, batch, cfg)`.

    python train_plain2.py
"""
import math
import time
from pathlib import Path
import sys
import torch
from einops import rearrange
from stable_pretraining.data.dataset_stats import ImageNet

from module import SIGReg
from train_no_hydra import build_loaders, build_model, load_cfg


def to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def inverse_transform(pixels):
    """Undo utils.get_img_preprocessor: normalized float (..., 3, H, W) -> uint8 (..., H, W, 3).

    Forward was: uint8 / 255 -> (x - ImageNet mean) / ImageNet std -> Resize(img_size).
    Exact when img_size equals the stored frame size (224 for pusht: Resize is a no-op);
    otherwise you get the resized frame back.
    """
    mean = pixels.new_tensor(ImageNet["mean"]).view(3, 1, 1)
    std = pixels.new_tensor(ImageNet["std"]).view(3, 1, 1)
    x = (pixels.float() * std + mean) * 255
    return x.round().clamp(0, 255).to(torch.uint8).movedim(-3, -1)


def pack_frame_skip(action, pixels, k):
    """Group k consecutive raw actions into one token and keep every k-th frame.

    action: (B, n*k + 1, a), pixels: (B, n*k + 1, C, H, W)
    ->  action: (B, n, k*a),  pixels: (B, n + 1, C, H, W)
    """
    n = min(action.size(1), pixels.size(1) - 1) // k
    action = rearrange(action[:, : n * k], "b (n k) a -> b n (k a)", k=k)
    pixels = pixels[:, : n * k + 1 : k]
    return action, pixels


def jepa_losses(model, sigreg, batch, cfg):
    """Predict future frame embeddings from (first frame, action prefix). Returns losses."""
    k = int(cfg.wm.action_frame_skip)
    last_only = bool(cfg.wm.dense_supervision.get("supervise_last_state_only", False))

    # 1. inputs: n action tokens a_1..a_n and n+1 frames x_0..x_n.
    #    n is the same for the whole batch (the sampler buckets samples by n).
    action = torch.nan_to_num(batch["action"], 0.0)  # missing actions -> 0
    action, pixels = pack_frame_skip(action, batch["pixels"], k)
    if last_only:
        pixels = pixels[:, [0, -1]]  # only x_0 and x_n are needed

    # 2. frames -> embeddings z: ViT CLS token -> projector MLP  (= JEPA.encode)
    B = pixels.size(0)
    frames = rearrange(pixels.float(), "b t c h w -> (b t) c h w")
    cls = model.encoder(frames, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    emb = rearrange(model.projector(cls), "(b t) d -> b t d", b=B)  # (B, n+1, D)

    # 3. action prefixes: causal transformer over [z_0, a_1..a_n] (module.py:89).
    #    act_emb[:, t] summarizes z_0 and a_1..a_(t+1), never later actions.
    act_emb = model.action_encoder(action, latent=emb[:, :1])  # (B, n, D)

    # 4. predict z_1..z_n from z_0 alone + each action prefix. The predictor is
    #    pointwise: every timestep is predicted independently, no attention.
    #    The targets z_1..z_n are never shown to the predictor.
    if last_only:
        act_emb, target = act_emb[:, -1:], emb[:, -1:]
    else:
        target = emb[:, 1:]
    pred = model.predictor(emb[:, 0], act_emb)  # (B, n, hidden)
    pred = model.pred_proj(rearrange(pred, "b t d -> (b t) d"))  # BatchNorm MLP needs 2D
    pred = rearrange(pred, "(b t) d -> b t d", b=B)  # (B, n, D)

    # 5. losses
    pred_loss = (pred - target).pow(2).mean()  # MSE in embedding space
    # SIGReg keeps each frame's embeddings (across the batch) close to an isotropic
    # Gaussian. Without it the encoder could collapse to a constant and get pred_loss = 0.
    sigreg_loss = sigreg(emb.transpose(0, 1))  # expects (T, B, D)
    loss = pred_loss + cfg.loss.sigreg.weight * sigreg_loss
    return {"loss": loss, "pred_loss": pred_loss, "sigreg_loss": sigreg_loss}


def main():
    cfg = load_cfg("pusht")
    run_dir = Path("results") / f"plain2-{cfg.subdir}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. data (must run before build_model: fills cfg.wm.action_dim etc.)
    train, val = build_loaders(cfg)

    # 2. model + loss regularizer (SIGReg has buffers only, nothing to train)
    model = build_model(cfg).to(device)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)

    # 3. optimizer + LR schedule (1% linear warmup, cosine to 0, stepped per batch)
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

    # 4. loop
    for epoch in range(cfg.trainer.max_epochs):
        model.train()
        t0 = time.perf_counter()
        
        num_iters_per_epoch = len(train)
        for step, batch in enumerate(train):
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = jepa_losses(model, sigreg, batch, cfg)

            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clip_val)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            if step % 1 == 0:
                
                td = time.perf_counter() - t0

                tps = td / (step + 1)
                elapsed_time = tps * (num_iters_per_epoch - step)

                ltt = out['loss'].item()
                lsg = out['sigreg_loss'].item()

                txt = '\rIter: [%d/%d] Losses: [%.2f / %.2f] Time: [%.2f/%.2f] ' % (
                    step, num_iters_per_epoch, ltt, lsg, td, elapsed_time
                    )
                sys.stdout.write(txt)
                sys.stdout.flush()
                # print(
                #     f"epoch {epoch} step {step}/{len(train)} "
                #     f"loss {out['loss'].item():.4f} pred {out['pred_loss'].item():.4f} "
                #     f"sigreg {out['sigreg_loss'].item():.4f} "
                #     f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s"
                # )

        model.eval()
        val_losses = []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            for batch in val:
                out = jepa_losses(model, sigreg, to_device(batch, device), cfg)
                val_losses.append(out["loss"].item())
        print(f"epoch {epoch} val loss {sum(val_losses) / len(val_losses):.4f}")

        torch.save(model, run_dir / f"{cfg.output_model_name}_epoch_{epoch + 1}_object.ckpt")


if __name__ == "__main__":
    main()
