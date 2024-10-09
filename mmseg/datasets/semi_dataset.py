# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import json
import os.path as osp

import mmcv
import numpy as np
import torch

from . import CityscapesDataset
from .builder import DATASETS


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class SemiDataset(object):

    def __init__(self, source, target_label,target_unlabel, cfg):
        self.source = source
        self.target_label = target_label
        self.target_unlabel = target_unlabel
        self.ignore_index = source.ignore_index
        self.CLASSES = source.CLASSES
        self.PALETTE = source.PALETTE
        # assert target.ignore_index == source.ignore_index
        # assert target.CLASSES == source.CLASSES
        # assert target.PALETTE == source.PALETTE

        # rcs_cfg = cfg.get('rare_class_sampling')
        # self.rcs_enabled = rcs_cfg is not None
        # if self.rcs_enabled:
        #     self.rcs_class_temp = rcs_cfg['class_temp']
        #     self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
        #     self.rcs_min_pixels = rcs_cfg['min_pixels']

        #     self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
        #         cfg['source']['data_root'], self.rcs_class_temp)
        #     mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
        #     mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

        #     with open(
        #             osp.join(cfg['source']['data_root'],
        #                      'samples_with_class.json'), 'r') as of:
        #         samples_with_class_and_n = json.load(of)
        #     samples_with_class_and_n = {
        #         int(k): v
        #         for k, v in samples_with_class_and_n.items()
        #         if int(k) in self.rcs_classes
        #     }
        #     self.samples_with_class = {}
        #     for c in self.rcs_classes:
        #         self.samples_with_class[c] = []
        #         for file, pixels in samples_with_class_and_n[c]:
        #             if pixels > self.rcs_min_pixels:
        #                 self.samples_with_class[c].append(file.split('/')[-1])
        #         assert len(self.samples_with_class[c]) > 0
        #     self.file_to_idx = {}
        #     for i, dic in enumerate(self.source.img_infos):
        #         file = dic['ann']['seg_map']
        #         if isinstance(self.source, CityscapesDataset):
        #             file = file.split('/')[-1]
        #         self.file_to_idx[file] = i

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        try:
            i1 = self.file_to_idx[f1]
            s1 = self.source[i1]
            if self.rcs_min_crop_ratio > 0:
                for j in range(10):
                    n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                    # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                    if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                        break
                    # Sample a new random crop from source image i1.
                    # Please note, that self.source.__getitem__(idx) applies the
                    # preprocessing pipeline to the loaded image, which includes
                    # RandomCrop, and results in a new crop of the image.
                    s1 = self.source[i1]
            i2 = np.random.choice(range(len(self.target)))
            s2 = self.target[i2]

            return {
                **s1, 'target_img_metas': s2['img_metas'],
                'target_img': s2['img']
            }
        except BaseException as e:
            print(i1)

    def __getitem__(self, idx):

        # if self.rcs_enabled:
        #     return self.get_rare_class_sample()
        # else:
        #     s1 = self.source[idx // len(self.target)]
        #     s2 = self.target[idx % len(self.target)]
        
        a = len(self.source)
        b = len(self.target_label)


        s1 = self.source[idx]
        # s11 = self.source[idx]
        s2 = self.target_unlabel[idx]   
        # s22 = self.target_unlabel[idx]    
        s3 = self.target_label[idx]  
        # s33 = self.target_label[idx]  

        # return {
        #     # **s1,
        #     'img': [s1['img'],s11['img']],
        #     'img_metas': [s1['img_metas'],s11['img_metas']],
        #     'gt_semantic_seg': [s1['gt_semantic_seg'],s11['gt_semantic_seg']],
        #      'target_label_img_metas': [s3['img_metas'],s33['img_metas']],
        #     'target_label_img': [s3['img'],s33['img']] , 
        #     'target_gt': [s3['gt_semantic_seg'],s33['gt_semantic_seg']],
        #      'target_unlabel_img_metas': [s2['img_metas'],s22['img_metas']],
        #     'target_unlabel_img': [s2['img'],s22['img']],


        # }

        return {
            **s1,
             'target_label_img_metas': s3['img_metas'],
            'target_label_img': s3['img'] ,
            'target_gt': s3['gt_semantic_seg'],
             'target_unlabel_img_metas': s2['img_metas'],
            'target_unlabel_img': s2['img'],
            

        }


    def __len__(self):
        return len(self.source) #* (len(self.target_unlabel) + len(self.target_label))
