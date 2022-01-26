_base_ = [
    '../_base_/models/cascade_mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=498,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=498,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=498,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 800), (512, 800), (544, 800), (576, 800),
                                 (608, 800), (640, 800), (672, 800), (704, 800),
                                 (736, 800), (768, 800), (800, 800)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(
                     type='Shear',
                     prob=0.4,
                     level=1),
                 dict(
                     type='Translate',
                     prob=0.3,
                     level=2),
                 dict(
                    type='Rotate',
                    prob=0.6,
                    level=6),
                 dict(
                    type='EqualizeTransform',
                    prob=0.5),
             ],
             [
                 dict(type='Resize',
                      img_scale=[(480, 800), (512, 800), (544, 800), (576, 800),
                                 (608, 800), (640, 800), (672, 800), (704, 800),
                                 (736, 800), (768, 800), (800, 800)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(
                    type='Rotate',
                    prob=0.5,
                    level=7),
                 dict(
                    type='EqualizeTransform',
                    prob=0.3),
             ],
             [
                 dict(type='Resize',
                      img_scale=[(480, 800), (512, 800), (544, 800), (576, 800),
                                 (608, 800), (640, 800), (672, 800), (704, 800),
                                 (736, 800), (768, 800), (800, 800)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(
                    type='ContrastTransform',
                     prob=0.3,
                     level=3
                 ),
                 dict(
                    type='BrightnessTransform',
                     prob=0.3,
                     level=3
                 ),
                 dict(
                     type='Shear',
                     prob=0.4,
                     level=1),
                 dict(
                     type='Translate',
                     prob=0.2,
                     level=5),
                 dict(
                    type='Rotate',
                    prob=0.6,
                    level=10),
                 dict(
                    type='EqualizeTransform',
                    prob=0.2),
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 800), (512, 800), (544, 800),
                                 (576, 800), (608, 800), (640, 800),
                                 (672, 800), (704, 800), (736, 800),
                                 (768, 800), (800, 800)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 800), (512, 800), (544, 800),
                                 (576, 800), (608, 800), (640, 800),
                                 (672, 800), (704, 800), (736, 800),
                                 (768, 800), (800, 800)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True),
                 dict(
                    type='ContrastTransform',
                     prob=0.3,
                     level=3
                 ),
                 dict(
                    type='BrightnessTransform',
                     prob=0.3,
                     level=3
                 ),
                 dict(
                     type='Shear',
                     prob=0.4,
                     level=1),
                 dict(
                    type='Rotate',
                    prob=0.6,
                    level=5),
                 dict(
                    type='EqualizeTransform',
                    prob=0.4),
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(640, 640), (864, 864)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(train=dict(pipeline=train_pipeline),
           val=dict(pipeline=test_pipeline),
           test=dict(pipeline=test_pipeline))
evaluation = dict(interval=2, metric=['bbox', 'segm'])
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
