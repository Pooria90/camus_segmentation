#!/bin/bash

python main.py --output_dir ./results/unets_4/ --image_size 256 --train_batch_size 16 --num_train_epochs 100 \
--model unet_small  --aug_train_frames "./data/generated-frames-edge-1-350000-fid/" --aug_train_masks ./data/train_seg_aug  --aug_prop 0.5 \
--valid_batch_size 8 --train_frames ./data/train_frames/ --train_masks ./data/train_seg/ \
--valid_frames ./data/valid_frames/ --valid_masks ./data/valid_seg/ \
--val_interval 4 --num_classes 4 --lr 0.001 \
--gpu_id 1
