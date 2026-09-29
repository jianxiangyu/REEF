<div align="center">

# REEF: Relation-Aware Graph Foundation Model

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-blue)](<[paper-url]()>)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange)](https://huggingface.co/ffjasonyu/REEF)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

**Relations are what graph foundation models need.** REEF treats each *relation type* as the fundamental transferable unit, so a single pre-trained model can transfer across datasets, domains, and tasks.

## 🔥 News
- *2026.09*: 🎉 REEF is accepted by **NeurIPS 2026**! 🥳
- *2026.09*: 🤗 Pretrained weights are released on [Hugging Face]([https://huggingface.co/<HF_REPO>](https://huggingface.co/ffjasonyu/REEF))!

## ❓ What is REEF

REEF is a **graph foundation model (GFM)** that treats **relations as the fundamental transferable unit** of graphs. Instead of learning at the node or dataset level, REEF builds a *relation vocabulary*: each relation type (e.g., a citation link, a molecular bond, a knowledge-graph predicate) is encoded from its textual description by a sentence encoder, and three hypernetworks turn these relation embeddings into:

1. a **relation-specific aggregator** for message passing,
2. a **relation-specific classifier** for downstream prediction, and
3. a **dataset-specific projector + feature bias** that adapts to different feature distributions.

This lets a single pre-trained model transfer across datasets, domains, and tasks (node classification, link prediction) in zero-shot and few-shot settings.

## 📂 Repository Structure

```
REEF/
├── code/                     # All source code
│   ├── main.py               # Pre-training entry point
│   ├── finetune.py           # Transfer / few-shot fine-tuning entry point
│   ├── downstream.py         # Downstream evaluation (node cls. & link pred.)
│   ├── model.py              # REEF model (GFM) definition
│   ├── layer.py              # Hypernetwork-generated GNN layers
│   ├── llm_model.py          # Sentence encoder (relation/description → embedding)
│   ├── pretrain_data.py      # Pre-training subgraph dataset builder
│   ├── dataset.py            # Dataset reading & SVD feature standardization
│   ├── utils.py              # Utility functions
│   ├── args.py               # All hyperparameters and settings
│   ├── relation_info.json    # Relation & dataset textual descriptions
│   ├── data_process/         # Per-domain raw-data loaders (KG, citation, ...)
│   ├── pretrained_data/      # Cached pre-training tensors (info.pt)  ← downloaded
│   └── pretrained_result/    # Pre-trained checkpoints (.pth)         ← downloaded
├── data/                     # Raw datasets
├── cache_data/               # Cached constructed subgraphs & LM cache
└── requirements.txt
```
## ⚡️ Installation

REEF is tested with **Python 3.10** and **PyTorch 1.13.1 (CUDA 11.7)**.

**1. Core dependencies**
```bash
conda create -n reef python=3.10 -y
conda activate reef

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.43.2 sentence-transformers==3.1.1
```

**2. PyG dependencies** (matched to `torch-1.13.1+cu117`)
```bash
pip install torch-geometric==2.6.1
pip install pyg-lib==0.4.0 torch-scatter==2.1.0 torch-sparse==0.6.15 \
    torch-cluster==1.6.0 torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

**3. Everything else**
```bash
pip install -r requirements.txt
```

> The relation/dataset descriptions are encoded with Sentence-BERT (`all-MiniLM-L6-v2`), downloaded automatically on first run into `cache_data/model`.

## 📊 Data

Raw datasets live under `data/` and cached subgraphs under `cache_data/`. The 10 datasets span four domains — knowledge graphs (FB15K237, WN18RR), citation networks (Cora, Citeseer, Pubmed), web pages (Cornell, Texas, Wisconsin), and co-purchase graphs (Photo, Computers). Per-domain loaders are in `code/data_process/`.
## 🤗 Pretrained Weights

Pretrained weights are hosted on Hugging Face: **https://huggingface.co/ffjasonyu/REEF**

Download the two files into `code/`:
```bash
# pip install huggingface_hub   (already in requirements.txt)
huggingface-cli download <HF_REPO> reef_pretrained.pth --local-dir ./code --repo-type model
huggingface-cli download <HF_REPO> edge_texts2id.json  --local-dir ./code --repo-type model
```

| File | Size | Description |
|---|---|---|
| `reef_pretrained.pth` | ~677 MB | Pretrained `state_dict` of the `GFM` model. |
| `edge_texts2id.json`  | ~30 KB  | Relation-text → id mapping (254 relations). |

> **Note.** `finetune.py` loads the checkpoint from `pretrained_result/<run-id>/Epoch_<n>.pth` and expects a cached `pretrained_data/<datasets>/info.pt` (precomputed relation embeddings). Place `reef_pretrained.pth` accordingly (or update the load path at the bottom of `finetune.py`); `info.pt` is regenerated automatically on the first pre-processing run.

## 🚀 Usage

All commands are run from inside the `code/` directory.

### Pre-training
```bash
cd code
python main.py --gpu 0
```
Checkpoints, the relation-id mapping, and per-epoch results are written to `code/pretrained_result/<timestamp>/`.

### Transfer / Few-shot Fine-tuning
```bash
cd code
python finetune.py --gpu 0
```
Fine-tuning loads a pre-trained checkpoint. Before running, set the checkpoint you want to load at the bottom of `finetune.py`:
```python
starttime   = '2025-02-14 22:43:34.927159'  # the pretrained_result/<run-id> folder
epoch_times = '0'                            # loads Epoch_0.pth
```
The target dataset(s) are controlled by `--transfer_tasks` in `args.py` (default: `cora`, node classification), with `--shot` few-shot labels per class.


## 🔎 Citation

If you find REEF useful, please cite:

```bibtex
@article{yu2025relation,
  title={Relation-Aware Graph Foundation Model},
  author={Yu, Jianxiang and Zhu, Jiapeng and Qian, Hao and Liu, Ziqi and Zhang, Zhiqiang and Li, Xiang},
  journal={arXiv preprint arXiv:2505.12027},
  year={2025}
}
```

## 📬 Contact

For questions or suggestions, feel free to open an issue or reach out to the authors.
