# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Config and checkpoint update
# - Saving instead of showing prediction

import os
from argparse import ArgumentParser

import mmcv
from tools.test import update_legacy_cfg

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_classes, get_palette
import cv2
import numpy as np
import glob, os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from demo.validation import *

class EvalDataset(Dataset):
    def __init__(self):
        self.label_path = ''
        self.data_path = ''
        
        self.label_list = sorted(glob.glob(self.label_path + '*.*'))
        self.data_list = sorted(glob.glob(self.data_path + '*.*'))
        
    def __getitem__(self, idx):
        label

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile(args.config)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        args.checkpoint,
        device=args.device,
        classes=get_classes(args.palette),
        palette=get_palette(args.palette),
        revise_checkpoint=[(r'^module\.', ''), ('model.', '')])

    # setting test data path
    root = '/mnt/workspace/dingyiming/Codes/semi-supervised-uda/data/DSEC_Semantic_e2vid_online/gt_fine/train_pro'
    label_path = "./output/"

    select_path = None
    os.makedirs(label_path)

    # find root data
    if isinstance(select_path, str):
        file_list = list()
        with open(label_path, 'r') as f:
            for file_name in f.readlines():
                file_list.append(os.path.join(root, file_name))
    else:
        file_list = sorted(glob.glob(root + '/*.*'))

    # init evaluator
    evaluator = Evaluator(11)

    for file_path in tqdm(file_list):
        file_name = file_path.split('/')[-1]
        if file_name.split('.')[-1] == 'npy':
            file_name = file_name.split('.')[0] + '.png'
        result = inference_segmentor(model, file_path)
        cv2.imwrite(label_path + file_name, result)


if __name__ == '__main__':
    main()
    