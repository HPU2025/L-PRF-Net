# L-PRF Net: Identifying Unsafe Behaviors in Laboratory Settings

This repository contains the official PyTorch implementation of our paper: **"L-PRF Net: A hybrid architecture for identifying unsafe behaviors in laboratory settings"**.

## 💡 Overview

L-PRF Net is a highly efficient video-based action recognition network designed specifically for monitoring and identifying unsafe behaviors in laboratory environments. To balance computational efficiency and recognition accuracy, this architecture introduces two key components:

*   **SGTP (Semantic-Guided Token Pruning)**: Reduces computational redundancy by pruning uninformative tokens based on semantic guidance.
*   **FFCM (Fine-Grained Feature Compensation Module)**: Compensates for spatial and temporal information loss caused by the pruning process, ensuring high-precision action recognition.

## 🛠️ Requirements and Environment

This code is built upon the [OpenMMLab](https://openmmlab.com/) ecosystem. 

*   **OS**: Linux (Ubuntu recommended)
*   **Python**: >= 3.8
*   **PyTorch**: >= 1.10.0
*   **Dependencies**: 
    *   `mmcv` 
    *   `mmaction2`
    *   `opencv-python`
    *   `matplotlib`
    *   `numpy`

**Hardware Setup:** All experiments and model training were conducted on a single **NVIDIA GeForce RTX 3080 Ti (12GB VRAM)**.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required packages:
```bash
git clone [https://github.com/YourUsername/L-PRF-Net.git](https://github.com/YourUsername/L-PRF-Net.git)
cd L-PRF-Net
pip install -r requirements.txt
