# [cite_start]L-PRF Net: An Efficient Hybrid Architecture for Identifying Unsafe Behaviors in Laboratory Settings [cite: 7]

[cite_start]This repository contains the official PyTorch implementation of our paper: **"L-PRF Net: A hybrid architecture for identifying unsafe behaviors in laboratory settings"**[cite: 1].

## 💡 Abstract

[cite_start]Video action recognition in complex controlled environments (e.g., laboratories and chemical plants) generally faces challenges of high spatio-temporal redundancy and difficulties in fine-grained feature extraction[cite: 14]. [cite_start]Due to the adoption of dense computing paradigms and static network topologies, existing hybrid video architectures suffer from redundant computational overhead and limited perception capabilities for subtle key actions[cite: 15]. 

[cite_start]To address these issues, we propose an efficient hybrid vision architecture, **L-PRF Net**, to optimize the trade-off between recognition accuracy and computational overhead[cite: 16].

## 🚀 Key Components

[cite_start]L-PRF Net is built upon the UniFormerV2 baseline [cite: 101] [cite_start]and introduces three collaborative modules[cite: 48]:

* [cite_start]**SGTP (Semantics-guided Token Pruning):** Introduced at the shallow feature level, it utilizes a cross-modal vision-language model (CLIP) as prior knowledge to filter out background noise while preserving the 3D spatio-temporal topology[cite: 16]. [cite_start]It effectively reduces computational overhead with a token retention ratio of 0.6[cite: 194].
* [cite_start]**ADRA (Adaptive Dynamic Routing Agent):** Designed during the deep interaction stage to achieve conditional computation via a lightweight gating mechanism[cite: 17]. [cite_start]It dynamically determines the activation depth of global aggregation blocks based on the semantic complexity of action samples[cite: 17].
* [cite_start]**FFCM (Fine-grained Feature Compensation Module):** Embedded to compensate for local information attenuation caused by sparsification[cite: 18]. [cite_start]It utilizes a dense dilated convolution array (with dilation rates r=1, 2, 3, 4) and a parallel dual attention mechanism (channel and spatial) to enhance the feature representation capability for subtle actions [cite: 18] [cite_start][cite: 153].

## 📊 Main Results

L-PRF Net achieves state-of-the-art trade-offs between accuracy and computational cost (GFLOPs).

| Dataset | Backbone | Views | GFLOPs | Params (M) | Top-1 Acc (%) | Top-5 Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [cite_start]**Kinetics-400** [cite: 212] | ViT-B | 4×3 | 157.3 | 133.5 | **85.6** | **96.9** |
| [cite_start]**Something-Something V2** [cite: 226] | ViT-B | 1×3 | 208.7 | 178.1 | **71.8** | **93.5** |
| [cite_start]**LabUB V1** (Ours) [cite: 239] | ViT-B | 4×3 | 213.3 | 125.7 | **80.3** | **95.2** |

## 🛠️ Installation & Environment

This project is based on the [OpenMMLab](https://openmmlab.com/) ecosystem, specifically `mmaction2`. All experiments were conducted on a single NVIDIA GeForce RTX 3080 Ti (12GB VRAM).

**Requirements:**
* OS: Linux
* Python >= 3.8
* PyTorch >= 1.10.0
* `mmcv`, `mmaction2`
* `opencv-python`, `matplotlib`, `numpy`

**Installation Steps:**
```bash
conda create -n lprf python=3.8 -y
conda activate lprf
# Install PyTorch
pip install torch torchvision torchaudio
# Install MMCV and MMAction2
pip install -U openmim
mim install mmcv
mim install mmaction2
# Clone this repo
git clone [https://github.com/YourUsername/L-PRF-Net.git](https://github.com/YourUsername/L-PRF-Net.git)
cd L-PRF-Net
pip install -r requirements.txt
