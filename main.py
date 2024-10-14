import os
import time
import pickle
import random
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
#from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from copy import deepcopy

from models import get_attention_unet, get_unet_large, get_unet_small, count_parameters
from data_utils import CustomTransform, SegDataset
from arguments import parse_args

def main(args):
    start_time = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    if args.augmentations:
        train_trans = CustomTransform()
    else:
        train_trans = CustomTransform(size=args.image_size,angle=0,translate=0,scale=0,shear=0,b_factor=0,c_factor=0,hflip=0)
        
    valid_trans = CustomTransform(size=args.image_size,angle=0,translate=0,scale=0,shear=0,b_factor=0,c_factor=0,hflip=0)

    train_ds = SegDataset(
        args.train_frames,
        args.train_masks,
        aug_image_dir=args.aug_train_frames,
        aug_mask_dir=args.aug_train_masks,
        aug_prop=args.aug_prop,
        transform=train_trans
    )
    valid_ds = SegDataset(args.valid_frames, args.valid_masks, transform=valid_trans)
    print(f'Training data size: {train_ds.__len__()}')
    print(f'Validation data size: {valid_ds.__len__()}')

    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    valid_dl = DataLoader(valid_ds, batch_size=args.valid_batch_size, pin_memory=torch.cuda.is_available())

    if args.model == 'unet_small':
        model = get_unet_small()
        model = model.to(device)
    elif args.model == 'unet_large':
        model = get_unet_large()
        model = model.to(device)
    elif args.model == 'attention_unet':
        model = get_attention_unet()
        model = model.to(device)
    
    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))
        print('Model loaded from {}'.format(args.load_model_path))
    
    print (f'Number of training parameters in the model: {count_parameters(model)}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.num_train_epochs,eta_min=0.0001)

    logs = {
        'epoch': [],
        'loss': [],
        'dice_mean': [],
        'dice_std': [],
        'dice_metric': [],
        'val_loss': [],
        # test_loss, test_dice_mean
    }

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    with open(args.output_dir + 'args.pkl','wb') as f:
        pickle.dump(args, f)

    # -------------------------------------------------
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1

    #loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.05,0.25,0.35,0.35]).to(device)) # W
    loss_func = DiceCELoss(include_background=False,softmax=True)
    dice_metric = DiceMetric(include_background=False,reduction="mean")

    iter = args.num_train_epochs
    for ep in range(iter):
        print("-" * 10)
        print(f"epoch {ep + 1}/{iter}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch in train_dl:
            step += 1

            xb, yb = batch[0].to(device), batch[1].to(device)
            yb = F.one_hot(yb.long(), args.num_classes).squeeze().permute(0, 3, 1, 2)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_func(y_pred, yb.float())
            loss.backward()

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_dl.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        print(f"epoch {ep + 1} average loss: {epoch_loss:.4f}")

        if (ep + 1) % val_interval == 0:
            logs['loss'].append(epoch_loss)
            logs['epoch'].append(ep + 1)
            model.eval()
            with torch.no_grad():
                step = 0
                dice_scores = []
                for batch in valid_dl:
                    step += 1

                    xb, yb = batch[0].to(device), batch[1].to(device)
                    yb = F.one_hot(yb.long(), args.num_classes).squeeze().permute(0, 3, 1, 2)

                    y_pred = model(xb)
                    loss = loss_func(y_pred, yb.float())
                    epoch_loss += loss.item()

                    # may need to change this part
                    #y_pred = torch.cat([torch.unsqueeze(post_trans(i),dim=0) for i in decollate_batch(y_pred)])
                    val_outputs = F.one_hot(torch.argmax(y_pred,dim=1), num_classes=args.num_classes).permute(0,3,1,2)
                    metric = dice_metric(val_outputs, yb)
                    dice_scores.append(metric)

                epoch_loss /= step
                logs['val_loss'].append(epoch_loss)

                dice_scores = torch.cat(dice_scores, dim=0)
                s = tuple(torch.std(dice_scores,dim=0).cpu().numpy())
                m = tuple(torch.mean(dice_scores,dim=0).cpu().numpy())
                logs['dice_mean'].append(m)
                logs['dice_std'].append(s)
                epoch_metric = torch.mean(dice_scores).item()
                logs['dice_metric'].append(epoch_metric)

                if epoch_metric > best_metric:
                    best_metric = epoch_metric
                    best_metric_epoch = ep + 1
                    torch.save(model.state_dict(), args.output_dir + f"best_metric_model_segmentation2d.pth")
                    print("saved new best metric model")

                print("current epoch: {} current dice loss: {:.4f} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            ep + 1, epoch_loss, epoch_metric, best_metric, best_metric_epoch))
                print(f"class mean dice: {m} class std dice: {s}")

                with open(args.output_dir + "logs.pkl", 'wb') as f:
                    pickle.dump(logs, f)
    end_time = time.time()
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    print("Elapsed time: {:.2f} mins.".format((end_time-start_time)/60))
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
