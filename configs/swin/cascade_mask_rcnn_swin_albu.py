_base_ = [
    '../_base_/models/cascade_mask_rcnn_swin_fpn.py',
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

albu_train_transforms = [
    dict(
        type="OneOf",
        transforms=[
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=1024,
                        width=1024,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=864,
                                width=864,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=832,
                                width=832,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=800,
                                width=800,
                                p=1.0
                            )
                        ], p=1.0
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=992,
                        width=992,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=864,
                                width=864,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=832,
                                width=832,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=800,
                                width=800,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=768,
                                width=768,
                                p=1.0
                            )
                        ], p=1.0
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=960,
                        width=960,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=864,
                                width=864,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=832,
                                width=832,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=800,
                                width=800,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=768,
                                width=768,
                                p=1.0
                            )
                        ], p=1.0
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=928,
                        width=928,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=864,
                                width=864,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=832,
                                width=832,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=800,
                                width=800,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=768,
                                width=768,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=736,
                                width=736,
                                p=1.0
                            )
                        ], p=1.0
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=896,
                        width=896,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=864,
                                width=864,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=832,
                                width=832,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=800,
                                width=800,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=768,
                                width=768,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=736,
                                width=736,
                                p=1.0
                            )
                        ], p=1.0
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=864,
                        width=864,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=832,
                                width=832,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=800,
                                width=800,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=768,
                                width=768,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=736,
                                width=736,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=704,
                                width=704,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=832,
                        width=832,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=800,
                                width=800,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=768,
                                width=768,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=736,
                                width=736,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=704,
                                width=704,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=800,
                        width=800,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=672,
                                width=672,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=768,
                                width=768,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=736,
                                width=736,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=704,
                                width=704,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=768,
                        width=768,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=672,
                                width=672,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=640,
                                width=640,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=736,
                                width=736,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=704,
                                width=704,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=736,
                        width=736,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=672,
                                width=672,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=640,
                                width=640,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=608,
                                width=608,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=704,
                                width=704,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=704,
                        width=704,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=672,
                                width=672,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=640,
                                width=640,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=576,
                                width=576,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=608,
                                width=608,
                                p=1.0
                            ),
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=672,
                        width=672,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=640,
                                width=640,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=576,
                                width=576,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=608,
                                width=608,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=544,
                                width=544,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=640,
                        width=640,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=512,
                                width=512,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=576,
                                width=576,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=608,
                                width=608,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=544,
                                width=544,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=608,
                        width=608,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=512,
                                width=512,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=480,
                                width=480,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=576,
                                width=576,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=544,
                                width=544,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=576,
                        width=576,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=512,
                                width=512,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=480,
                                width=480,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=544,
                                width=544,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=544,
                        width=544,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=512,
                                width=512,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=480,
                                width=480,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=448,
                                width=448,
                                p=1.0
                            ),
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=512,
                        width=512,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=480,
                                width=480,
                                p=1.0
                            ),
                            dict(
                                type='RandomCrop',
                                height=448,
                                width=448,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type="Sequential",
                transforms = [
                    dict(
                        type='Resize',
                        height=480,
                        width=480,
                        p=1.0
                    ),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='RandomCrop',
                                height=448,
                                width=448,
                                p=1.0
                            )
                        ]
                    )
                ]
            ),
            dict(
                type='Resize',
                height=448,
                width=448,
                p=1.0
            ),
        ],
        p=1.0
    ),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0725,
        scale_limit=0.125,
        rotate_limit=30,
        interpolation=2,
        p=0.4),

    dict(
        type='OneOf',
        transforms=[
            dict(type='RandomGamma', p=1.0),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=1.0
            )
        ],
        p=0.2),
     dict(
        type='OneOf',
        transforms=[
            dict(type='CLAHE', p=1.0),
            dict(
                type='RGBShift',
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=10,
                sat_shift_limit=35,
                val_shift_limit=25,
                p=1.0),
        ],
        p=0.2),
    dict(
        type="HorizontalFlip",
        p=0.5
    ),
    dict(
        type='VerticalFlip', p=0.25),
    dict(type='RandomRotate90', p=0.3),
    dict(type='Cutout', p=0.4),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=5, p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
            dict(type='GaussNoise', var_limit=25, p=1.0),
            dict(type='JpegCompression', quality_lower=80, quality_upper=95, p=1.0),
        ],
        p=0.3),

]


# augmentation strategy originates from DETR / Sparse RCNN
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='AutoAugment',
#          policies=[
#              [
#                  dict(type='Resize',
#                       img_scale=[(480, 960), (512, 960), (544, 960), (576, 960),
#                                  (608, 960), (640, 960), (672, 960), (704, 960),
#                                  (736, 960), (768, 960), (800, 960)],
#                       multiscale_mode='value',
#                       keep_ratio=True)
#              ],
#              [
#                  dict(type='Resize',
#                       img_scale=[(400, 1333), (500, 1333), (600, 1333)],
#                       multiscale_mode='value',
#                       keep_ratio=True),
#                  dict(type='RandomCrop',
#                       crop_type='absolute_range',
#                       crop_size=(384, 600),
#                       allow_negative_crop=True),
#                  dict(type='Resize',
#                       img_scale=[(480, 960), (512, 960), (544, 960), (576, 960),
#                                  (608, 960), (640, 960), (672, 960), (704, 960),
#                                  (736, 960), (768, 960), (800, 960)],
#                       multiscale_mode='value',
#                       override=True,
#                       keep_ratio=True)
#              ]
#          ]),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)

# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

dataset_type = 'CocoDataset'
data_root = '../data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.3,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(480, 480), (640, 640), (960, 960)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes =  ['bread-wholemeal',
           'jam',
           'water',
           'bread-sourdough',
           'banana',
           'soft-cheese',
           'ham-raw',
           'hard-cheese',
           'cottage-cheese',
           'bread-half-white',
           'coffee-with-caffeine',
           'fruit-salad',
           'pancakes',
           'tea',
           'salmon-smoked',
           'avocado',
           'spring-onion-scallion',
           'ristretto-with-caffeine',
           'ham',
           'egg',
           'bacon-frying',
           'chips-french-fries',
           'juice-apple',
           'chicken',
           'tomato-raw',
           'broccoli',
           'shrimp-boiled',
           'beetroot-steamed-without-addition-of-salt',
           'carrot-raw',
           'chickpeas',
           'french-salad-dressing',
           'pasta-hornli',
           'sauce-cream',
           'meat-balls',
           'pasta',
           'tomato-sauce',
           'cheese',
           'pear',
           'cashew-nut',
           'almonds',
           'lentils',
           'mixed-vegetables',
           'peanut-butter',
           'apple',
           'blueberries',
           'cucumber',
           'cocoa-powder',
           'greek-yaourt-yahourt-yogourt-ou-yoghourt',
           'maple-syrup-concentrate',
           'buckwheat-grain-peeled',
           'butter',
           'herbal-tea',
           'mayonnaise',
           'soup-vegetable',
           'wine-red',
           'wine-white',
           'green-bean-steamed-without-addition-of-salt',
           'sausage',
           'pizza-margherita-baked',
           'salami',
           'mushroom',
           'bread-meat-substitute-lettuce-sauce',
           'tart',
           'tea-verveine',
           'rice',
           'white-coffee-with-caffeine',
           'linseeds',
           'sunflower-seeds',
           'ham-cooked',
           'bell-pepper-red-raw',
           'zucchini',
           'green-asparagus',
           'tartar-sauce',
           'lye-pretzel-soft',
           'cucumber-pickled',
           'curry-vegetarian',
           'yaourt-yahourt-yogourt-ou-yoghourt-natural',
           'soup-of-lentils-dahl-dhal',
           'soup-cream-of-vegetables',
           'balsamic-vinegar',
           'salmon',
           'salt-cake-vegetables-filled',
           'bacon',
           'orange',
           'pasta-noodles',
           'cream',
           'cake-chocolate',
           'pasta-spaghetti',
           'black-olives',
           'parmesan',
           'spaetzle',
           'salad-lambs-ear',
           'salad-leaf-salad-green',
           'potatoes-steamed',
           'white-cabbage',
           'halloumi',
           'beetroot-raw',
           'bread-grain',
           'applesauce-unsweetened-canned',
           'cheese-for-raclette',
           'mushrooms',
           'bread-white',
           'curds-natural-with-at-most-10-fidm',
           'bagel-without-filling',
           'quiche-with-cheese-baked-with-puff-pastry',
           'soup-potato',
           'bouillon-vegetable',
           'beef-sirloin-steak',
           'taboule-prepared-with-couscous',
           'eggplant',
           'bread',
           'turnover-with-meat-small-meat-pie-empanadas',
           'mungbean-sprouts',
           'mozzarella',
           'pasta-penne',
           'lasagne-vegetable-prepared',
           'mandarine',
           'kiwi',
           'french-beans',
           'tartar-meat',
           'spring-roll-fried',
           'pork-chop',
           'caprese-salad-tomato-mozzarella',
           'leaf-spinach',
           'roll-of-half-white-or-white-flour-with-large-void',
           'pasta-ravioli-stuffing',
           'omelette-plain',
           'tuna',
           'dark-chocolate',
           'sauce-savoury',
           'dried-raisins',
           'ice-tea',
           'kaki',
           'macaroon',
           'smoothie',
           'crepe-plain',
           'chicken-nuggets',
           'chili-con-carne-prepared',
           'veggie-burger',
           'cream-spinach',
           'cod',
           'chinese-cabbage',
           'hamburger-bread-meat-ketchup',
           'soup-pumpkin',
           'sushi',
           'chestnuts',
           'coffee-decaffeinated',
           'sauce-soya',
           'balsamic-salad-dressing',
           'pasta-twist',
           'bolognaise-sauce',
           'leek',
           'fajita-bread-only',
           'potato-gnocchi',
           'beef-cut-into-stripes-only-meat',
           'rice-noodles-vermicelli',
           'tea-ginger',
           'tea-green',
           'bread-whole-wheat',
           'onion',
           'garlic',
           'hummus',
           'pizza-with-vegetables-baked',
           'beer',
           'glucose-drink-50g',
           'chicken-wing',
           'ratatouille',
           'peanut',
           'high-protein-pasta-made-of-lentils-peas',
           'cauliflower',
           'quiche-with-spinach-baked-with-cake-dough',
           'green-olives',
           'brazil-nut',
           'eggplant-caviar',
           'bread-pita',
           'pasta-wholemeal',
           'sauce-pesto',
           'oil',
           'couscous',
           'sauce-roast',
           'prosecco',
           'crackers',
           'bread-toast',
           'shrimp-prawn-small',
           'panna-cotta',
           'romanesco',
           'water-with-lemon-juice',
           'espresso-with-caffeine',
           'egg-scrambled-prepared',
           'juice-orange',
           'ice-cubes',
           'braided-white-loaf',
           'emmental-cheese',
           'croissant-wholegrain',
           'hazelnut-chocolate-spread-nutella-ovomaltine-caotina',
           'tomme',
           'water-mineral',
           'hazelnut',
           'bacon-raw',
           'bread-nut',
           'black-forest-tart',
           'soup-miso',
           'peach',
           'figs',
           'beef-filet',
           'mustard-dijon',
           'rice-basmati',
           'mashed-potatoes-prepared-with-full-fat-milk-with-butter',
           'dumplings',
           'pumpkin',
           'swiss-chard',
           'red-cabbage',
           'spinach-raw',
           'naan-indien-bread',
           'chicken-curry-cream-coconut-milk-curry-spices-paste',
           'crunch-muesli',
           'biscuits',
           'bread-french-white-flour',
           'meatloaf',
           'fresh-cheese',
           'honey',
           'vegetable-mix-peas-and-carrots',
           'parsley',
           'brownie',
           'dairy-ice-cream',
           'tea-black',
           'carrot-cake',
           'fish-fingers-breaded',
           'salad-dressing',
           'dried-meat',
           'chicken-breast',
           'mixed-salad-chopped-without-sauce',
           'feta',
           'praline',
           'tea-peppermint',
           'walnut',
           'potato-salad-with-mayonnaise-yogurt-dressing',
           'kebab-in-pita-bread',
           'kolhrabi',
           'alfa-sprouts',
           'brussel-sprouts',
           'bacon-cooking',
           'gruyere',
           'bulgur',
           'grapes',
           'pork-escalope',
           'chocolate-egg-small',
           'cappuccino',
           'zucchini-stewed-without-addition-of-fat-without-addition-of-salt',
           'crisp-bread-wasa',
           'bread-black',
           'perch-fillets-lake',
           'rosti',
           'mango',
           'sandwich-ham-cheese-and-butter',
           'muesli',
           'spinach-steamed-without-addition-of-salt',
           'fish',
           'risotto-without-cheese-cooked',
           'milk-chocolate-with-hazelnuts',
           'cake-oblong',
           'crisps',
           'pork',
           'pomegranate',
           'sweet-corn-canned',
           'flakes-oat',
           'greek-salad',
           'cantonese-fried-rice',
           'sesame-seeds',
           'bouillon',
           'baked-potato',
           'fennel',
           'meat',
           'bread-olive',
           'croutons',
           'philadelphia',
           'mushroom-average-stewed-without-addition-of-fat-without-addition-of-salt',
           'bell-pepper-red-stewed-without-addition-of-fat-without-addition-of-salt',
           'white-chocolate',
           'mixed-nuts',
           'breadcrumbs-unspiced',
           'fondue',
           'sauce-mushroom',
           'tea-spice',
           'strawberries',
           'tea-rooibos',
           'pie-plum-baked-with-cake-dough',
           'potatoes-au-gratin-dauphinois-prepared',
           'capers',
           'vegetables',
           'bread-wholemeal-toast',
           'red-radish',
           'fruit-tart',
           'beans-kidney',
           'sauerkraut',
           'mustard',
           'country-fries',
           'ketchup',
           'pasta-linguini-parpadelle-tagliatelle',
           'chicken-cut-into-stripes-only-meat',
           'cookies',
           'sun-dried-tomatoe',
           'bread-ticino',
           'semi-hard-cheese',
           'margarine',
           'porridge-prepared-with-partially-skimmed-milk',
           'soya-drink-soy-milk',
           'juice-multifruit',
           'popcorn-salted',
           'chocolate-filled',
           'milk-chocolate',
           'bread-fruit',
           'mix-of-dried-fruits-and-nuts',
           'corn',
           'tete-de-moine',
           'dates',
           'pistachio',
           'celery',
           'white-radish',
           'oat-milk',
           'cream-cheese',
           'bread-rye',
           'witloof-chicory',
           'apple-crumble',
           'goat-cheese-soft',
           'grapefruit-pomelo',
           'risotto-with-mushrooms-cooked',
           'blue-mould-cheese',
           'biscuit-with-butter',
           'guacamole',
           'pecan-nut',
           'tofu',
           'cordon-bleu-from-pork-schnitzel-fried',
           'paprika-chips',
           'quinoa',
           'kefir-drink',
           'm-m-s',
           'salad-rocket',
           'bread-spelt',
           'pizza-with-ham-with-mushrooms-baked',
           'fruit-coulis',
           'plums',
           'beef-minced-only-meat',
           'pizza-with-ham-baked',
           'pineapple',
           'soup-tomato',
           'cheddar',
           'tea-fruit',
           'rice-jasmin',
           'seeds',
           'focaccia',
           'milk',
           'coleslaw-chopped-without-sauce',
           'pastry-flaky',
           'curd',
           'savoury-puff-pastry-stick',
           'sweet-potato',
           'chicken-leg',
           'croissant',
           'sour-cream',
           'ham-turkey',
           'processed-cheese',
           'fruit-compotes',
           'cheesecake',
           'pasta-tortelloni-stuffing',
           'sauce-cocktail',
           'croissant-with-chocolate-filling',
           'pumpkin-seeds',
           'artichoke',
           'champagne',
           'grissini',
           'sweets-candies',
           'brie',
           'wienerli-swiss-sausage',
           'syrup-diluted-ready-to-drink',
           'apple-pie',
           'white-bread-with-butter-eggs-and-milk',
           'savoury-puff-pastry',
           'anchovies',
           'tuna-in-oil-drained',
           'lemon-pie',
           'meat-terrine-pate',
           'coriander',
           'falafel-balls',
           'berries',
           'latte-macchiato-with-caffeine',
           'faux-mage-cashew-vegan-chers',
           'beans-white',
           'sugar-melon',
           'mixed-seeds',
           'hamburger',
           'hamburger-bun',
           'oil-vinegar-salad-dressing',
           'soya-yaourt-yahourt-yogourt-ou-yoghourt',
           'chocolate-milk-chocolate-drink',
           'celeriac',
           'chocolate-mousse',
           'cenovis-yeast-spread',
           'thickened-cream-35',
           'meringue',
           'lamb-chop',
           'shrimp-prawn-large',
           'beef',
           'lemon',
           'croque-monsieur',
           'chives',
           'chocolate-cookies',
           'birchermuesli-prepared-no-sugar-added',
           'fish-crunchies-battered',
           'muffin',
           'savoy-cabbage-steamed-without-addition-of-salt',
           'pine-nuts',
           'chorizo',
           'chia-grains',
           'frying-sausage',
           'french-pizza-from-alsace-baked',
           'chocolate',
           'cooked-sausage',
           'grits-polenta-maize-flour',
           'gummi-bears-fruit-jellies-jelly-babies-with-fruit-essence',
           'wine-rose',
           'coca-cola',
           'raspberries',
           'roll-with-pieces-of-chocolate',
           'goat-average-raw',
           'lemon-cake',
           'coconut-milk',
           'rice-wild',
           'gluten-free-bread',
           'pearl-onions',
           'buckwheat-pancake',
           'bread-5-grain',
           'light-beer',
           'sugar-glazing',
           'tzatziki',
           'butter-herb',
           'ham-croissant',
           'corn-crisps',
           'lentils-green-du-puy-du-berry',
           'cocktail',
           'rice-whole-grain',
           'veal-sausage',
           'cervelat',
           'sorbet',
           'aperitif-with-alcohol-aperol-spritz',
           'dips',
           'corn-flakes',
           'peas',
           'tiramisu',
           'apricots',
           'cake-marble',
           'lamb',
           'lasagne-meat-prepared',
           'coca-cola-zero',
           'cake-salted',
           'dough-puff-pastry-shortcrust-bread-pizza-dough',
           'rice-waffels',
           'sekt',
           'brioche',
           'vegetable-au-gratin-baked',
           'mango-dried',
           'processed-meat-charcuterie',
           'mousse',
           'sauce-sweet-sour',
           'basil',
           'butter-spread-puree-almond',
           'pie-apricot-baked-with-cake-dough',
           'rusk-wholemeal',
           'beef-roast',
           'vanille-cream-cooked-custard-creme-dessert',
           'pasta-in-conch-form',
           'nuts',
           'sauce-carbonara',
           'fig-dried',
           'pasta-in-butterfly-form-farfalle',
           'minced-meat',
           'carrot-steamed-without-addition-of-salt',
           'ebly',
           'damson-plum',
           'shoots',
           'bouquet-garni',
           'coconut',
           'banana-cake',
           'waffle',
           'apricot-dried',
           'sauce-curry',
           'watermelon-fresh',
           'sauce-sweet-salted-asian',
           'pork-roast',
           'blackberry',
           'smoked-cooked-sausage-of-pork-and-beef-meat-sausag',
           'bean-seeds',
           'italian-salad-dressing',
           'white-asparagus',
           'pie-rhubarb-baked-with-cake-dough',
           'tomato-stewed-without-addition-of-fat-without-addition-of-salt',
           'cherries',
           'nectarine']
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annotations.json',
        img_prefix=data_root + 'train/images/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annotations.json',
        img_prefix=data_root + 'val/images/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val_annotations.json',
        img_prefix=data_root + 'val/images/',
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
