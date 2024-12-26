#!/bin/bash
cd ..
python main.py --output_dir ./results/unetatt_real_1/ --image_size 256 --train_batch_size 8 --num_train_epochs 100 \
--model attention_unet \
--valid_batch_size 8 --train_frames ./data/train_frames/ --train_masks ./data/train_seg/ \
--valid_frames ./data/valid_frames/ --valid_masks ./data/valid_seg/ \
--val_interval 1 --num_classes 4 --lr 0.001 \
--gpu_id 1
