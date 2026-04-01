# dataset settings
data_keys = ['motion', 'motion_mask', 'motion_length']
meta_keys = ['text']
train_pipeline = [
    dict(
        type='Normalize',
        mean_path='data/datasets/human_ml3d/mean_std/Mean.npy',
        std_path='data/datasets/human_ml3d/mean_std/Std.npy'),
    dict(type='Crop', crop_size=196),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='TextMotionDataset',
            dataset_name='human_ml3d',
            data_prefix='data',
            pipeline=train_pipeline,
            ann_file='split/train.txt',
            motion_dir='motion_data',
            text_dir='texts',
        ),
        times=200
    ),
    test=dict(
        type='TextMotionDataset',
        dataset_name='human_ml3d',
        data_prefix='data',
        pipeline=train_pipeline,
        ann_file='split/test.txt',
        motion_dir='motion_data',
        text_dir='texts',
        eval_cfg=None,
        test_mode=False
    )
)
