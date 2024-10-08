# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (460, 640)
dsec_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(1280, 720)),
    # dict(type='RGB2Gray'),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RGB2Gray', out_channels=1),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size,cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='SemiDataset',
        source=dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            split=None,
            pipeline=cityscapes_train_pipeline),
        target_label=dict(
            type='DSEC_E2vidDataset',
            data_root='data/DSEC_Semantic_e2vid_online',
            img_dir='event/train',
            ann_dir='gt_fine/train_pro',
            split = 'history_txt/desc_label_2.txt',
            pipeline=dsec_train_pipeline),
        target_unlabel=dict(
            type='DSEC_E2vidDataset',
            data_root='data/DSEC_Semantic_e2vid_online',
            img_dir='event/train',
            ann_dir='gt_fine/train_pro',
            split = 'history_txt/desc_unlabel_2.txt',
            pipeline=dsec_train_pipeline)),
    val=dict(
        type='DSEC_E2vidDataset',
        data_root='data/DSEC_Semantic_e2vid_online',
        img_dir='event/val',
        ann_dir='gt_fine/val',
        split = None,
        pipeline=test_pipeline),
    test=dict(
        type='DSEC_E2vidDataset',
        data_root='data/DSEC_Semantic_e2vid_online',
        img_dir='event/val',
        ann_dir='gt_fine/val',
        split = None,
        pipeline=test_pipeline))


