<h1 align="center">Fast LeWorldModel — Plain PyTorch Re-implementation</h1>

<p align="center">
  <a href="https://fast-lewm.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://github.com/Yuntian-Gao/Fast-LeWorldModel">
    <img src="https://img.shields.io/badge/Official_Code-GitHub-181717?logo=github&logoColor=white" alt="Official Code">
  </a>
  <a href="https://arxiv.org/abs/2606.26217">
    <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper">
  </a>
</p>

This repository contains a **plain PyTorch re-implementation** of
**Fast LeWorldModel**, written for research purposes by **Yura Vasiliuk** at
GlobalLogic, part of Hitachi Digital.

**This is not the official code.** Fast LeWorldModel was created by
[Yuntian Gao](https://yuntian-gao.github.io/) and
[Xiangyu Xu](https://xuxy09.github.io/) (corresponding author). Their official
implementation is here:
**[github.com/Yuntian-Gao/Fast-LeWorldModel](https://github.com/Yuntian-Gao/Fast-LeWorldModel)**.

All credit for the method, the model and the original codebase goes to the
original authors. If you use this work, please cite their paper (see
[Citation](#citation)).

## What this re-implementation changes

The official code trains with PyTorch Lightning and `stable_pretraining`, and
it uses Hydra for configs. Evaluation is handed to `stable_worldmodel`'s
planning tools. This repository keeps the original model, data pipeline and
configs. It adds Hydra-free scripts that write the training loop and the
planner out in plain PyTorch:

| File | What it does |
|------|--------------|
| `train_no_hydra.py` | The original Lightning training, with no Hydra. Settings are edited in `load_cfg`. |
| `train_plain.py` | Plain PyTorch training loop (no Lightning). It reuses `lejepa_forward` from `train.py`. |
| `train_plain2.py` | Plain PyTorch training loop with the JEPA forward pass and losses written out in full. |
| `eval_plain.py` | Plain planning evaluation (MPC + CEM + cost written out). `stable_worldmodel` is only used to create the environments. |
| `train_pt.ipynb`, `eval_pt.ipynb` | Notebooks for running the plain-PyTorch training and evaluation step by step. |

The plain training loops follow what the original Lightning setup does:
bf16 autocast, gradient-norm clipping at 1.0, AdamW, and 1% linear warmup
followed by cosine decay. They leave out multi-GPU training, W&B logging,
resumable checkpoints and deterministic algorithms.

The original `train.py` and `eval.py` are still included, so you can compare
the two versions.

## Installation

```bash
conda create -n fast-lewm python=3.10 -y
conda activate fast-lewm
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

Use the same datasets and layout as
[LeWM](https://github.com/lucas-maes/le-wm#data). Put the extracted `.h5` files
in one directory:

```text
stablewm_data/
├── pusht_expert_train.h5
├── tworoom_expert_train.h5
├── reacher_expert_train.h5
└── cube_expert_train.h5
```

Then set the `STABLEWM_HOME` environment variable:

```bash
export STABLEWM_HOME=/absolute/path/to/stablewm_data
```

`eval_plain.py` reads data from `~/stablewm_data` by default. To use another
location, change `cfg.cache_dir` in its `load_cfg`.

## Pretrained Checkpoints

The official Fast-LeWorldModel checkpoints are on Hugging Face:
[naiverer/fast-leworldmodel](https://huggingface.co/naiverer/fast-leworldmodel).

## Training (plain PyTorch)

Settings (batch size, workers, frame skip and so on) are set in `load_cfg` in
`train_no_hydra.py`. The default dataset is `pusht`.

```bash
python train_plain2.py   # forward pass written out; checkpoints in results/plain2-<timestamp>/
python train_plain.py    # reuses lejepa_forward; checkpoints in outputs/plain-<timestamp>/
```

Each epoch saves a full model object as
`Fast-lewm_epoch_<N>_object.ckpt`.

## Planning and Evaluation (plain PyTorch)

```bash
python eval_plain.py path/to/Fast-lewm_epoch_N_object.ckpt [env]
```

`env` defaults to `pusht`. Metrics and rollout videos go to
`outputs/eval-<env>-<timestamp>/`. The CEM and evaluation budgets are set in
`load_cfg` in `eval_plain.py`.

## Original (Lightning + Hydra) scripts

The original commands still work. See the
[official README](https://github.com/Yuntian-Gao/Fast-LeWorldModel) for details.

```bash
python train.py data=pusht
python eval.py --config-name=pusht \
  eval.dataset_path=/absolute/path/to/pusht_expert_train.h5 \
  eval.ckpt_path=/absolute/path/to/Fast-lewm_pusht_object.ckpt
```

## Acknowledgements

This work is based entirely on the official
[Fast-LeWorldModel](https://github.com/Yuntian-Gao/Fast-LeWorldModel) code by
Yuntian Gao and Xiangyu Xu. Their code is in turn built on
[LeWorldModel](https://github.com/lucas-maes/le-wm). Thanks to both teams of
authors for releasing their code. The original MIT license is kept in
[LICENSE](LICENSE).

## Citation

If you find this work useful, please cite the original paper:

```bibtex
@article{gao2026fast,
  title={Fast LeWorldModel},
  author={Gao, Yuntian and Xu, Xiangyu},
  journal={arXiv preprint arXiv:2606.26217},
  year={2026}
}
```
