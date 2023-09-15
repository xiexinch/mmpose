_base_ = ['../../../_base_/default_runtime.py']

val_cfg = None
test_cfg = None

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=120, val_interval=10)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01))

# learning policy
param_scheduler = [
    dict(type='ExponentialLR', gamma=0.99, end=60, by_epoch=True)
]

auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=20),
)

# codec settings
train_codec = dict(
    type='MotionBERTLabel', num_keypoints=133, concat_vis=True, mode='train')
val_codec = dict(
    type='MotionBERTLabel', num_keypoints=133, concat_vis=True, rootrel=True)

# model settings
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='DSTFormer',
        in_channels=3,
        feat_size=512,
        depth=5,
        num_keypoints=133,
        num_heads=8,
        mlp_ratio=2,
        seq_len=243,
        att_fuse=True,
    ),
    head=dict(
        type='MotionRegressionHead',
        in_channels=512,
        out_channels=3,
        embedding_size=512,
        loss=dict(type='MPJPEVelocityJointLoss'),
        decoder=val_codec,
    ),
    test_cfg=dict(flip_test=True),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/'
        'pose_lift/h36m/motionbert_pretrain_h36m-29ffebf5_20230719.pth'),
)

# base dataset settings
dataset_type = 'H36MWholeBodyDataset'
data_root = 'data/h36m/'

# pipelines
train_pipeline = [
    dict(type='GenerateTarget', encoder=train_codec),
    dict(
        type='RandomFlipAroundRoot',
        keypoints_flip_cfg=dict(center_mode='static', center_x=0.),
        target_flip_cfg=dict(center_mode='static', center_x=0.),
        flip_label=True),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'factor', 'camera_param'))
]

# data loaders
train_dataloader = dict(
    batch_size=32,
    prefetch_factor=4,
    pin_memory=True,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='json/RGBto3D_train.json',
        seq_len=1,
        multiple_target=151,
        multiple_target_step=81,
        camera_param_file='processed/annotation_body3d/cameras.pkl',
        data_root=data_root,
        data_prefix=dict(img='processed/images/'),
        pipeline=train_pipeline,
    ))
