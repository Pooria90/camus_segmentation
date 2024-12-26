#!/bin/bash

python segmentation-v5.py --output_dir ./results/v5-run-73b/ --image_size 256 --train_batch_size 8 --num_train_epochs 200 \
--valid_batch_size 8 --train_frames ./data/train-frames-aug-63/ --train_masks ./data/train-masks-aug-63/ \
--valid_frames ./data/valid-frames-efn/ --valid_masks ./data/valid-masks-efn/ \
--val_interval 1 --num_classes 4 --lr 0.001 --gpu_id 1 \
--load_model_path ./results/v5-run-73/best_metric_model_segmentation2d.pth
