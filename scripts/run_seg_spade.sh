#!/bin/bash
cd ..
python main.py --output_dir ./results/unetatt_spade_3_50p/ --image_size 256 --train_batch_size 8 --num_train_epochs 200 \
--model attention_unet \
--aug_train_frames "./data/spade_iter50_train/" \
--aug_train_masks ./data/train_seg_aug  --aug_prop 0.5 \
--valid_batch_size 8 --train_frames ./data/train_frames/ --train_masks ./data/train_seg/ \
--valid_frames ./data/valid_frames/ --valid_masks ./data/valid_seg/ \
--val_interval 1 --num_classes 4 --lr 0.001 \
--gpu_id 1
