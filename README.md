# HPL_ESS

HPL-ESS: Hybrid Pseudo-Labeling for Unsupervised Event-based Semantic Segmentation
[CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Jing_HPL-ESS_Hybrid_Pseudo-Labeling_for_Unsupervised_Event-based_Semantic_Segmentation_CVPR_2024_paper.pdf)


We test our codes under the following environment:

- Ubuntu 20.04
- NVIDIA Driver: 515.65.01
- CUDA 11.0
- Python 3.8.19
- PyTorch 1.7.0

To start:

1. Clone this repository.

2. Install packages

   ~~~
   conda create -n ess python=3.8 -y
   conda activate ess
   pip install --upgrade pip 
   pip install -r requirements.txt
   # * for training
   
   pip install ninja
   pip install flash-attn
   ~~~
