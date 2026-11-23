"""Plain planning evaluation: same procedure as eval.py, no Hydra, planner written out.

eval.py hands the work to stable_worldmodel: World.evaluate_from_dataset (env loop),
WorldModelPolicy (MPC action buffer), CEMSolver (planning) and JEPA.get_cost (cost).
Here those are spelled out; swm.World is only used to create the environments.

    python eval_plain.py path/to/Fast-lewm_epoch_N_object.ckpt

Skipped vs eval.py: per-env `seed` / `variation.*` dataset columns (pusht has none),
expert frames in the videos, the rollout-consistency cost (weight 0.0 in all configs).
"""
import os

os.environ["MUJOCO_GL"] = "egl"

import sys
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import stable_worldmodel as swm
import torch
from einops import rearrange
from omegaconf import OmegaConf
from stable_pretraining.data.dataset_stats import ImageNet

CONFIG_DIR = Path(__file__).parent / "config" / "eval"


def load_cfg(env: str = "pusht"):
    cfg = OmegaConf.load(CONFIG_DIR / f"{env}.yaml")
    cfg.pop("defaults")
    cfg.solver = OmegaConf.load(CONFIG_DIR / "solver" / "cem.yaml")

    # --- knobs ---------------------------------------------------------------
    cfg.cache_dir = os.path.expanduser("~/stablewm_data")
    cfg.eval.num_eval = 50  # episodes = parallel envs
    cfg.eval.goal_offset_steps = 25  # goal = expert frame this many steps after start
    cfg.eval.eval_budget = 50  # env steps allowed to reach the goal
    cfg.solver.num_samples = 300  # CEM candidates per iteration
    cfg.solver.n_steps = 30  # CEM iterations per replan
    cfg.solver.topk = 30  # CEM elites
    cfg.save_video = True
    return cfg


def to_model_pixels(frames, device):
    """uint8 (N, H, W, 3) -> normalized float (N, 3, H, W), as eval.img_transform.

    Inverse of train_plain2.inverse_transform. eval.py also resizes to eval.img_size,
    a no-op here because envs render at 224x224.
    """
    x = torch.as_tensor(frames, device=device).permute(0, 3, 1, 2).float() * (1.0 / 255)
    mean = x.new_tensor(ImageNet["mean"]).view(3, 1, 1)
    std = x.new_tensor(ImageNet["std"]).view(3, 1, 1)
    return (x - mean) / std


def encode(model, pixels):
    """(N, 3, H, W) -> (N, D): ViT CLS token -> projector  (= JEPA.encode)."""
    cls = model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    return model.projector(cls)


def predict_final(model, z0, tokens):
    """Roll the world model forward from z0 and return the last predicted embedding.

    z0: (N, D); tokens: (N, H, blocks, token_dim) normalized action tokens
    (H planning steps, each made of `blocks` tokens of `action_block_size` raw actions).
    """
    z = z0
    for t in range(tokens.size(1)):
        # causal transformer over [z, a_1..a_blocks]; keep only the full-prefix summary
        act = model.action_encoder(tokens[:, t], return_last_only=True, latent=z[:, None])
        z = model.pred_proj(model.predictor(z, act)[:, 0])  # (N, D)
    return z


def plan(model, z0, z_goal, cfg, gen, init=None):
    """Cross-entropy method over normalized action plans (= CEMSolver + JEPA.get_cost).

    Returns the final plan mean (N, H, plan_dim) and the mean elite cost per env.
    """
    N, device = z0.size(0), z0.device
    S, K = cfg.solver.num_samples, cfg.solver.topk
    H = cfg.plan_config.horizon
    blocks = cfg.plan_config.action_num_blocks
    plan_dim = cfg.act_dim * blocks * cfg.plan_config.action_block_size  # 2 * 5 * 5 = 50

    # Gaussian over plans: mean starts at 0 (or the previous plan's tail), std at var_scale
    mean = torch.zeros(N, H, plan_dim, device=device)
    if init is not None:
        mean[:, : init.size(1)] = init
    std = torch.full_like(mean, float(cfg.solver.var_scale))

    z0_s = z0.repeat_interleave(S, 0)  # one copy per candidate: (N*S, D)
    zg_s = z_goal.repeat_interleave(S, 0)
    for _ in range(cfg.solver.n_steps):
        cand = torch.randn(N, S, H, plan_dim, generator=gen, device=device)
        cand = cand * std[:, None] + mean[:, None]
        cand[:, 0] = mean  # always keep the current mean as a candidate

        tokens = rearrange(cand, "n s h (f a) -> (n s) h f a", f=blocks)
        z_pred = predict_final(model, z0_s, tokens)
        cost = (z_pred - zg_s).pow(2).sum(-1).view(N, S)  # squared distance to goal embedding

        elite_cost, idx = cost.topk(K, dim=1, largest=False)
        elite = cand[torch.arange(N, device=device)[:, None], idx]  # (N, K, H, plan_dim)
        mean, std = elite.mean(1), elite.std(1)
    return mean, elite_cost.mean(1)


def sample_start_goal_rows(dataset, cfg):
    """Pick num_eval start rows whose goal (start + goal_offset - 1) is in the same episode."""
    ep_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep = dataset.get_col_data(ep_col)
    step = dataset.get_col_data("step_idx")
    ep_len = np.zeros(ep.max() + 1, dtype=np.int64)
    np.maximum.at(ep_len, ep, step + 1)

    offset = cfg.eval.goal_offset_steps
    valid = np.nonzero(step <= ep_len[ep] - offset - 1)[0]
    rng = np.random.default_rng(cfg.seed)
    rows = np.sort(valid[rng.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
    # eval.py loads frames [start, start + offset) and uses the last one as goal
    return rows, rows + offset - 1


@torch.no_grad()
def main(ckpt_path, env="pusht"):
    cfg = load_cfg(env)
    N, budget = cfg.eval.num_eval, cfg.eval.eval_budget
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("outputs") / f"eval-{env}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. model (object checkpoint saved by ModelObjectCallBack / train_plain*.py)
    model = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.eval().requires_grad_(False)

    # 2. dataset: start/goal frames + action stats (planner works in normalized action space)
    dataset = swm.data.HDF5Dataset(cfg.eval.dataset_name, cache_dir=cfg.cache_dir)
    start_rows, goal_rows = sample_start_goal_rows(dataset, cfg)
    start = dataset.get_row_data(start_rows)
    goal = dataset.get_row_data(goal_rows)
    actions = dataset.get_col_data("action").astype(np.float64)
    actions = actions[~np.isnan(actions).any(1)]
    act_mean, act_std = actions.mean(0), actions.std(0)  # = sklearn StandardScaler
    cfg.act_dim = actions.shape[1]

    # 3. envs: reset, then put each env into its start state with its goal
    cfg.world.max_episode_steps = 2 * budget
    world = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(224, 224))
    world.reset(seed=None, options=[{} for _ in range(N)])
    values = {**start, **{f"goal_{k}": v for k, v in goal.items()}}
    for i, e in enumerate(world.envs.unwrapped.envs):
        for spec in cfg.eval.callables:  # e.g. _set_state(state=...), _set_goal_state(goal_state=...)
            kwargs = {
                name: values[arg.value][i].copy() if arg.get("in_dataset", True) else arg.value
                for name, arg in spec.args.items()
            }
            getattr(e.unwrapped, spec.method)(**kwargs)

    # 4. MPC loop: plan, execute receding_horizon * action_block env steps, replan
    gen = torch.Generator(device=device).manual_seed(int(cfg.solver.seed))
    z_goal = encode(model, to_model_pixels(goal["pixels"], device))  # goal is fixed
    obs = start["pixels"]  # first observation = dataset frame, as in evaluate_from_dataset
    R = cfg.plan_config.receding_horizon
    buffer, warm, plan_costs = [], None, []
    success = np.zeros(N, dtype=bool)
    frames = np.empty((N, budget + 1, *obs.shape[1:]), dtype=np.uint8)

    for t in range(budget):
        frames[:, t] = obs
        if not buffer:
            z0 = encode(model, to_model_pixels(obs, device))
            mean, cost = plan(model, z0, z_goal, cfg, gen, init=warm)
            plan_costs.append(cost.cpu().numpy())
            warm = mean[:, R:]  # warm start: unexecuted tail seeds the next plan
            buffer = list(rearrange(mean[:, :R], "n r (k a) -> (r k) n a", a=cfg.act_dim))

        action = buffer.pop(0).cpu().numpy() * act_std + act_mean  # un-normalize
        _, _, terminated, _, infos = world.envs.step(action.astype(np.float32))
        world.envs.unwrapped._autoreset_envs = np.zeros(N)  # keep envs alive after success
        success |= terminated
        obs = infos["pixels"][:, -1]
    frames[:, -1] = obs

    # 5. results
    metrics = {
        "success_rate": 100.0 * success.mean(),
        "episode_successes": success.astype(int).tolist(),
        "plan_cost_mean": float(np.mean(plan_costs)),
    }
    print(metrics)
    with open(run_dir / "results.txt", "w") as f:
        f.write(f"ckpt: {ckpt_path}\n{OmegaConf.to_yaml(cfg)}\nmetrics: {metrics}\n")

    if cfg.save_video:  # left: agent, right: goal
        for i in range(N):
            with imageio.get_writer(run_dir / f"rollout_{i}.mp4", fps=15, codec="libx264") as w:
                for frame in frames[i]:
                    w.append_data(np.hstack([frame, goal["pixels"][i]]))
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
