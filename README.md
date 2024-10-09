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
3. Unzip mmseg.zip   

### Data Preparation

#### DSEC data

Download the following compressed files of DSEC datasets here. They require about 200GB of storage space.

~~~
'zurich_city_00_a', 'zurich_city_01_a', 'zurich_city_02_a',
'zurich_city_04_a', 'zurich_city_05_a', 'zurich_city_06_a',
'zurich_city_07_a', 'zurich_city_08_a'
~~~

The directory should look like this:

    # seq_name (e.g. zurich_city_00_a)
    # ├── semantic
    # │   ├── left
    # │   │   ├── 11classes
    # │   │   │   └──data
    # │   │   │       ├── 000000.png
    # │   │   │       └── ...
    # │   │   └── 19classes
    # │   │       └──data
    # │   │           ├── 000000.png
    # │   │           └── ...
    # │   └── timestamps.txt
    # └── events
    #     └── left
    #         ├── events.h5
    #         └── rectify_map.h5

#### CityScape data

Download leftImg8bit_trainvaltest.zip and gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/) and extract them  `data/cityscapes`.

#### E2VID data

clone this [repository](https://github.com/uzh-rpg/rpg_e2vid) and run on DSEC datasets to reconstruct the event streams into simulated images, save to ```data/DSEC_Semantic_e2vid_offline```

#### Off-line sampling 
run sample_percentage.py offline and randomly extract part of the data (1/16) as hybird-label in training, save to ```data/DSEC_Semantic_e2vid_offline/unlabel6.txt```.

### Training

Firstly, download the MiT ImageNet weights (b3-b5) provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training) from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`. Further, download the checkpoint of our model and extract it to the folder `work_dirs/`.


 A training job can be launched using:

```
python run_experiments.py --config configs/xxformer/gta2dsec_e2vid_offline_semi_xxformer.py
```

### Tips:
All the training details are similar to [here](https://github.com/lhoyer/DAFormer?tab=readme-ov-file#daformer-improving-network-architectures-and-training-strategies-for-domain-adaptive-semantic-segmentation), only the files in the mmseg need to be changed, especially loading dataset and 'dacs' model file. The dataset processing, especially the event data, can refer to this [here](https://github.com/uzh-rpg/ess).
