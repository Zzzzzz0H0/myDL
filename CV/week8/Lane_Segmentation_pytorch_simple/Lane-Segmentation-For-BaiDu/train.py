import config as cfg
import pandas as pd

from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn

from utils.dataset import BasicDataset
from models.unet_base import unet_base
from eval import eval_net
from tqdm import tqdm
from dice_loss_1 import MulticlassDiceLoss

def main():
    # network = 'deeplabv3p'
    # save_model_path = "./model_weights/" + network + "_"
    # model_path = "./model_weights/" + network + "_0_6000"
    data_dir = '~/workspace/myDL/CV/week8/Lane_Segmentation_pytorch/data_list/train.csv'
    val_percent = .1
    
    epochs = 9

    dataset = BasicDataset(data_dir, img_size=cfg.IMG_SIZE, crop_offset = cfg.crop_offset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = unet_base(cfg.num_classes, cfg.IMG_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.base_lr, betas=(0.9, 0.99))

    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = MulticlassDiceLoss()

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if cfg.num_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = model(imgs)

                loss = bce_criterion(masks_pred, true_masks) + dice_criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])

                global_step += 1
                if global_step % (len(dataset) // (10 * cfg.batch_size)) == 0:
                    val_score = eval_net(model, val_loader, device, n_val)
                    print ('val_score:', val_score)
                    # if cfg.num_classes > 1:
                    #     logging.info('Validation cross entropy: {}'.format(val_score))
                    #     writer.add_scalar('Loss/test', val_score, global_step)

                    # else:
                    #     logging.info('Validation Dice Coeff: {}'.format(val_score))
                    #     writer.add_scalar('Dice/test', val_score, global_step)

                    # writer.add_images('images', imgs, global_step)
                    # if net.n_classes == 1:
                    #     writer.add_images('masks/true', true_masks, global_step)
                    #     writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)


main()