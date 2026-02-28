'''
train scipt for enhanced gan
- add self attention to generator
- add self attention to modality-specific feature extractor
- add cross modal attention for fusion
- add complementary loss terms including modality-specific segmentation loss and fused feature multiscale loss
'''

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models_bank.gan_mmformer import gan_enhanced
import torch.nn.functional as F
from models_bank.PotsdamDataset import PotsdamDataset_noNVDI
from models_bank.VaihingenDataset import VaihingenDataset
from models_bank import criterion_rs

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np
import yaml
import time
import wandb
from tqdm import tqdm
from contextlib import contextmanager
import platform
    
    
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


def save_model(best_loss, val_loss, saveModelPath, model, optimizer, lr_scheduler):
    if val_loss < best_loss:
        best_loss = val_loss
        
        dir = os.listdir(saveModelPath)
        full_path = [os.path.join(saveModelPath, x) for x in dir]
        if len(dir) >= 3:
            # remove oldest checkpoint b4 saving a new one
            oldest_file = min(full_path, key=os.path.getctime)
            os.remove(oldest_file)
        
        f_name = '_'.join([str(int(time.time())), f'loss{best_loss:.4f}'])
        best_model_path = os.path.join(saveModelPath, f_name)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_loss': best_loss
            },
            best_model_path
        )
        
    return best_loss


def cal_val_loss(dict_results, labels, num_cls=5):
    fuse_pred = dict_results['fuse_preds']
    fuse_cross_loss = criterion_rs.softmax_weighted_loss(fuse_pred, labels, num_cls=num_cls)
    fuse_dice_loss = criterion_rs.dice_loss(fuse_pred, labels, num_cls=num_cls)
    fuse_loss = fuse_cross_loss + fuse_dice_loss
    
    return fuse_loss


def validate(model, val_loader, device):
    running_val_loss = 0.0
    model.eval() # avoid error when batch has only 1 sample
    model.is_training = False

    with torch.no_grad():
        # assume that validation dataset only have one batch
        for val_imgs, val_labels, val_masks in val_loader:
            val_imgs = val_imgs.to(device)
            val_labels= torch.squeeze(val_labels, dim=1)
            val_labels = criterion_rs.expand_target(val_labels)
            val_labels = val_labels.type(torch.FloatTensor).to(device)
            val_masks = val_masks.to(device)

            # assume there is no full modality in validation dataset as well
            dict_val_pred = model(val_imgs, val_masks)
            val_loss = cal_val_loss(dict_val_pred, val_labels)
            
            running_val_loss += val_loss.item()

        # free gpu memory
        del val_imgs, val_labels, val_masks
        
    return running_val_loss / len(val_loader)


def train_log(dict_train_loss, val_loss, example_ct, log_file_path):
    train_loss = dict_train_loss['total_loss'].item()
    rgb_gan_loss = dict_train_loss['rgb_gan_loss']
    ndsm_gan_loss = dict_train_loss['ndsm_gan_loss']
    fuse_loss = dict_train_loss['fuse_loss']
    rgb_seg_loss = dict_train_loss['rgb_seg_loss']
    ndsm_seg_loss = dict_train_loss['ndsm_seg_loss']
    fuse_scale_loss = dict_train_loss['fuse_scale_loss']
    
    # val_img = dict_val_seg['img']
    # val_full_seg = dict_val_seg['full_seg']
    # val_miss_rgb_seg = dict_val_seg['miss_rgb_seg']
    # val_miss_ndsm_seg = dict_val_seg['miss_ndsm_seg']
    
    wandb.log({ 
        "train_loss": train_loss,
        "rgb_gan_loss": rgb_gan_loss,
        "ndsm_gan_loss": ndsm_gan_loss,
        "fuse_loss": fuse_loss,
        "rgb_seg_loss": rgb_seg_loss,
        "ndsm_seg_loss": ndsm_seg_loss,
        "fuse_scale_loss": fuse_scale_loss,
        "val_loss": val_loss
        # "val_img": val_img,
        # "val_full_seg": val_full_seg,
        # "val_miss_rgb_seg": val_miss_rgb_seg,
        # "val_miss_ndsm_seg": val_miss_ndsm_seg,
        }, step=example_ct)

    with open(log_file_path, 'a') as f:
        print(f"Train loss after {str(example_ct).zfill(5)} examples: {train_loss:.3f}", file=f)
        print(f"Val loss after {str(example_ct).zfill(5)} examples: {val_loss:.3f}", file=f)


@contextmanager
def memory_management():
    try:
        yield
    finally:
        # Only clear cache when memory usage is high
        if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.8:
            torch.cuda.empty_cache()
            

def cal_train_loss(dict_results, labels):
    rgb_g_loss = dict_results['rgb_g_loss']
    rgb_d_loss = dict_results['rgb_d_loss']
    rgb_gan_loss = rgb_g_loss + rgb_d_loss
    ndsm_g_loss = dict_results['ndsm_g_loss']
    ndsm_d_loss = dict_results['ndsm_d_loss']
    ndsm_gan_loss = ndsm_g_loss + ndsm_d_loss
    
    rgb_preds = dict_results['rgb_preds']
    ndsm_preds = dict_results['ndsm_preds']
    fuse_preds = dict_results['fuse_preds']
    fuse_scale_preds = dict_results['fuse_scale_preds']
    
    fuse_cross_loss = criterion_rs.softmax_weighted_loss(fuse_preds, labels)
    fuse_dice_loss = criterion_rs.dice_loss(fuse_preds, labels)
    fuse_loss = fuse_cross_loss + fuse_dice_loss
    
    rgb_cross_loss = criterion_rs.softmax_weighted_loss(rgb_preds, labels)
    rgb_dice_loss = criterion_rs.dice_loss(rgb_preds, labels)
    rgb_seg_loss = rgb_cross_loss + rgb_dice_loss
    
    ndsm_cross_loss = criterion_rs.softmax_weighted_loss(ndsm_preds, labels)
    ndsm_dice_loss = criterion_rs.dice_loss(ndsm_preds, labels)
    ndsm_seg_loss = ndsm_cross_loss + ndsm_dice_loss
    
    scale_cross_loss = torch.zeros(1).cuda().float()
    scale_dice_loss = torch.zeros(1).cuda().float()
    for scale in fuse_scale_preds:
        scale_cross_loss += criterion_rs.softmax_weighted_loss(scale, labels)
        scale_dice_loss += criterion_rs.dice_loss(scale, labels)
    fuse_scale_loss = scale_cross_loss + scale_dice_loss
    
    # TODO tuning weights for each loss term
    # in this case prioritize the fuse loss
    rgb_gan_loss = rgb_gan_loss * 0.05
    ndsm_gan_loss = ndsm_gan_loss * 0.05
    total_loss = rgb_gan_loss + ndsm_gan_loss + fuse_loss + rgb_seg_loss + ndsm_seg_loss + fuse_scale_loss

    if rgb_gan_loss != 0:
        rgb_gan_loss = rgb_gan_loss.item()
    if ndsm_gan_loss != 0:
        ndsm_gan_loss = ndsm_gan_loss.item()
    
    dict_loss_terms = {
        'rgb_gan_loss': rgb_gan_loss,
        'ndsm_gan_loss': ndsm_gan_loss,
        'fuse_loss': fuse_loss.item(),
        'rgb_seg_loss': rgb_seg_loss.item(),
        'ndsm_seg_loss': ndsm_seg_loss.item(),
        'fuse_scale_loss': fuse_scale_loss.item(),
        'total_loss': total_loss,
    }
    
    return dict_loss_terms


def train_batch(images, labels, masks, model, device):
    model.train()
    
    # Move tensors to device
    images = images.to(device)
    masks = masks.to(device)
    labels = torch.squeeze(labels, dim=1)
    labels = criterion_rs.expand_target(labels)
    labels = labels.type(torch.FloatTensor).to(device)
    
    model.is_training = True
    
    # Use context manager for memory management
    with memory_management():
        dict_results = model(images, masks)
        dict_losses = cal_train_loss(dict_results, labels)
    
    # Clear references
    del images, dict_results, labels, masks
    
    return dict_losses


def train(model, train_dataloader, val_dataloader, args, device, checkpoint=None):
    
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.99, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15)
    
    # resume training
    if checkpoint is not None:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

    # save best model logic
    best_loss = 100

    # Create folder to save checkpoints                                       
    saveModelPath = args.checkpoints_path
    os.makedirs(saveModelPath, exist_ok=True)

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, log="all", log_freq=100)

    # log file
    parent_dir = os.path.dirname(args.checkpoints_path)
    
    os_name = platform.system()
    if os_name == 'Windows':
        # window file system
        checkpoints_id = args.checkpoints_path.split('\\')[-1]
    else:
        # linux file system
        checkpoints_id = args.checkpoints_path.split('/')[-1]
    log_file_path = os.path.join(parent_dir, f'training_log_{checkpoints_id}.txt')
    
    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0
    mini_batch_size = 8

    for epoch in tqdm(range(args.epochs)):
        
        # TODO change this if train on Vaihingen dataset
        # randomly select a subset of images for training each epoch - 1200 images out of 4k1 images
        if epoch != 0:
            train_dataset = PotsdamDataset_noNVDI(args.train_data_path, amount=1200)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            
            del train_dataset
        
        batch_ct = 0
        
        for _, (images, labels, masks) in enumerate(train_dataloader):
            if len(images) < 2:
                continue # skip batch if it has less than 2 samples
            
            if (batch_ct % mini_batch_size) == 0 or (_+1) == len(train_dataloader):
                scenarios = [
                    [True, True],
                    [False, True],
                    [True, False],
                ]
                chosen_mask = random.choice(scenarios)
                masks = torch.tensor([chosen_mask] * images.size(0))
                
                with open(log_file_path, 'a') as f:
                    print(f"Chosen mask is {chosen_mask}", file=f)
            
            with open(log_file_path, 'a') as f:
                print(f"Epoch {epoch}, batch {batch_ct}, example {example_ct}", file=f)
                
            dict_losses = train_batch(images, labels, masks, model, device)
            
            example_ct +=  len(images)

            scaled_loss = dict_losses['total_loss'] / mini_batch_size
            scaled_loss.backward()
            
            if ((batch_ct % mini_batch_size) == 0) or (_+1) == len(train_dataloader):
                # Step with optimizer
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            del images, labels, masks
            
            if ((batch_ct + 1) % 32) == 0 or (_+1) == len(train_dataloader):
                with memory_management():
                    val_loss = validate(model, val_dataloader, device)

                    # samples = next(iter(val_dataloader))
                    # dict_val_seg = log_val_image(samples, model)

                    # train_log(dict_losses, val_loss, example_ct, dict_val_seg, log_file_path)
                    train_log(dict_losses, val_loss, example_ct, log_file_path)

                    # Save best model
                    best_loss = save_model(best_loss, val_loss, saveModelPath,
                            model, optimizer, lr_scheduler)
            
            batch_ct += 1
            
    return 0


if __name__ == '__main__':
    # Read configuration from config.yaml - window file system
    os_name = platform.system()
    if os_name == 'Windows':
        with open('experiments\\gan_mmformer\\config.yaml', 'r') as file:
                config = yaml.safe_load(file)
    else:
        # linux file system
        with open('experiments/gan_mmformer/config_ubuntu.yaml', 'r') as file: # linux file system
            config = yaml.safe_load(file)

    # Parse arguments from config
    parser = argparse.ArgumentParser(description='original mmformer vaihingen')
    parser.add_argument('--train_data_path', type=str, default=config['data']['train_data_path'])
    parser.add_argument('--val_data_path', type=str, default=config['data']['val_data_path'])
    parser.add_argument('--test_data_path', type=str, default=config['data']['test_data_path'])
    parser.add_argument('--batch_size', type=int, default=config['training']['batch_size'])
    parser.add_argument('--epochs', type=int, default=config['training']['epochs'])
    parser.add_argument('--checkpoints_path', type=str, default=config['callbacks']['checkpoints_path'])
    
    args = parser.parse_args()
    
    now = str(int(time.time()))
    checkpoints_pth = os.path.join(args.checkpoints_path, now)
    args.checkpoints_path = checkpoints_pth
    
    # train_dataset = VaihingenDataset(args.train_data_path, mode='shaspec')
    # val_dataset = VaihingenDataset(args.val_data_path, mode='shaspec')
    
    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    
    # train_dataset = PotsdamDataset(args.train_data_path, mode='shaspec')
    train_dataset = PotsdamDataset_noNVDI(args.train_data_path, amount=1200)
    val_dataset = PotsdamDataset_noNVDI(args.val_data_path)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    
    wandb.login()
    
    with wandb.init(project="paper_enhanced_gan_missing", entity="nhikieu"):
            # access all HPs through wandb.config, so logging matches execution!
            config = wandb.config
            config.checkpoints_path = args.checkpoints_path
            
            my_model = gan_enhanced.Model(num_cls=5)
            
            # resume_from_ckpt = r'/home/nknk/Documents/ML_Experiments/MissingModality/experiments/gan_mmformer/checkpoints/1746487774/1746492579_loss0.8767'
            # resume_from_ckpt = r"/home/nknk/Documents/ML_Experiments/MissingModality/experiments/gan_mmformer/checkpoints/1746665136/1746706107_loss0.7932"
            
            # potsdam checkpoint
            # resume_from_ckpt = r"D:\0_MissingModality\experiments\gan_mmformer\checkpoints\1747010393\1747055827_loss0.8477"
            
            # potsdam - best after the first 100 epochs
            # resume_from_ckpt = r"D:\0_MissingModality\experiments\gan_mmformer\checkpoints\1747148254_loss0.8483"

            # potsdam - continue to train on ubuntu due to memory crash on windows
            # resume_from_ckpt = r"/home/nknk/Documents/ML_Experiments/MissingModality/experiments/gan_mmformer/checkpoints/1747639482_loss0.8460"
            
            # 3rd round of training 50 epoch more from the last checkpoint after 2nd 100 epochs
            resume_from_ckpt = r"D:\0_MissingModality\experiments\gan_mmformer\checkpoints\1747741203_loss0.7375"

            train(my_model, train_dataloader, val_dataloader, args, 'cuda', checkpoint=resume_from_ckpt)
            # train(my_model, train_dataloader, val_dataloader, args, 'cuda')