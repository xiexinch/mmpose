_base_ = ['../_base_/default_runtime.py']

# runtime
max_epochs = 100
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=2023)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 210 to 420 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
train_codec = dict(
    type='Naive3DLabel',
    input_size=(192, 256, 192),
    simcc_split_ratio=2.0,
    sigma=(4.9, 5.66, 4.9),
    normalize=False)

val_codec = dict(
    type='Naive3DLabel',
    input_size=(192, 256, 192),
    simcc_split_ratio=2.0,
    sigma=(4.9, 5.66, 4.9),
    normalize=False,
    test_mode=True,
    gt_field='keypoints_3d_gt')

# model settings
model = dict(
    type='TopdownPoseEstimator3D',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'  # noqa
        )),
    head=dict(
        type='RTM3DHead',
        in_channels=768,
        out_channels=17,
        input_size=train_codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in train_codec['input_size']]),
        simcc_split_ratio=train_codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=val_codec),
    test_cfg=dict(flip_test=False))

# data settings
dataset_type = 'UBody3DCOCODataset'
data_mode = 'topdown'
data_root = 'data/UBody/'

backend_args = dict(backend='local')
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         f'{data_root}': 's3://openmmlab/datasets/pose/UBody/',
#         f'{data_root}': 's3://openmmlab/datasets/pose/UBody/'
#     }))

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine3D', input_size=train_codec['input_size']),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.),
        ]),
    dict(type='GenerateTarget', encoder=train_codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root', 'target_root_index', 'target_mean',
                   'target_std'))
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine3D', input_size=val_codec['input_size']),
    dict(type='GenerateTarget', encoder=val_codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('warp_mat', 'camera_param', 'z_max', 'z_min'))
]

scenes = [
    'Magic_show', 'Entertainment', 'ConductMusic', 'Online_class', 'TalkShow',
    'Speech', 'Fitness', 'Interview', 'Olympic', 'TVShow', 'Singing',
    'SignLanguage', 'Movie', 'LiveVlog', 'VideoConference'
]
skip_scenes = ['Speech', 'Movie']

train_datasets, val_datasets = [], []
for scene in scenes:
    train_ann = f'annotations/{scene}/train_3dkeypoint_annotation.json'
    val_ann = f'annotations/{scene}/val_3dkeypoint_annotation.json'
    train_dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_mode=data_mode,
        data_prefix=dict(img='images/'),
        pipeline=[
            dict(
                type='KeypointCombiner',
                num_keypoints=17,
                mapping=[((11, 12), 0), (12, 1), (14, 2), (16, 3), (11, 4),
                         (13, 5), (15, 6), ((5, 6, 11, 12), 7), ((5, 6), 8),
                         (0, 9), ((1, 2), 10), (5, 11), (7, 12), (9, 13),
                         (6, 14), (8, 15), (10, 16)],
                flip_indices=[
                    0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13
                ])
        ])
    train_datasets.append(train_dataset)
    if scene not in skip_scenes:
        val_dataset = train_dataset.copy()
        val_dataset.update({'ann_file': val_ann})
        val_datasets.append(val_dataset)

h36m_train_dataset = dict(
    type='H36MCOCODataset',
    ann_file='annotation_body2d/h36m_coco_train_fps50.json',
    data_root='data/h36m/',
    data_prefix=dict(img='images/'),
    camera_param_file='annotation_body3d/cameras.pkl',
    pipeline=[])
h36m_val_dataset = h36m_train_dataset.copy()
h36m_val_dataset.update({'ann_file': 'annotation_body2d/h36m_test_fps50.json'})

train_datasets.append(h36m_train_dataset)
val_datasets.append(h36m_val_dataset)

train_dataloader = dict(
    batch_size=256,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        datasets=train_datasets,
        pipeline=train_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/h36m.py'),
        test_mode=False))
val_dataloader = dict(
    batch_size=256,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='H36MCOCODataset',
        ann_file='annotation_body2d/h36m_test_fps50.json',
        data_root='data/h36m/',
        data_prefix=dict(img='images/'),
        camera_param_file='annotation_body3d/cameras.pkl',
        pipeline=val_pipeline))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]

# evaluators
val_evaluator = [
    dict(
        type='SimpleMPJPE',
        mode='mpjpe',
        pred_field='keypoints',
        gt_field='keypoints_3d_gt',
        gt_mask_field='keypoints_3d_visible'),
    dict(
        type='SimpleMPJPE',
        mode='p-mpjpe',
        pred_field='keypoints',
        gt_field='keypoints_3d_gt',
        gt_mask_field='keypoints_3d_visible')
]
test_evaluator = val_evaluator
