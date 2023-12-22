_base_ = ['../../../_base_/default_runtime.py']

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=100, val_interval=10)

# optimizer
optim_wrapper = dict(
    dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2))

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=100000,
        by_epoch=False)
]

auto_scale_lr = dict(base_batch_size=4096)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1))

codec = dict(
    type='ImagePoseLifting',
    num_keypoints=133,
    root_index=(11, 12),
    remove_root=False,
)

# model settings
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='LargeSimpleBaseline', in_channels=133 * 2, channels=1024),
    head=dict(
        type='TemporalRegressionHead',
        in_channels=1024,
        num_joints=133,
        loss=dict(type='MSELoss'),
        decoder=codec,
    ))

# base dataset settings
dataset_type = 'COCOWholebody3D'
data_root = 'data/coco/'

# pipelines
train_pipeline = [
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root', 'target_root_index', 'target_mean',
                   'target_std'))
]
val_pipeline = train_pipeline

# data loaders
train_dataloader = dict(
    batch_size=512,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='MSCOCO_train.npz',
        seq_len=1,
        causal=True,
        data_root=data_root,
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=512,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='H3WBDataset',
        ann_file='annotation_body3d/h3wb_train.npz',
        seq_len=1,
        causal=True,
        data_root='data/h36m/',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        train_mode=False,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='SimpleMPJPE', mode='mpjpe'),
    dict(type='SimpleMPJPE', mode='p-mpjpe')
]
test_evaluator = val_evaluator
