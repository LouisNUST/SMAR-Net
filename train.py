# -*- coding:utf-8 -*-
from __future__ import division, absolute_import, print_function

import argparse
import csv
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from path import Path
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import models
from utils import custom_transforms
from utils.utils import save_path_formatter, AverageMeter
from utils.loss import Img_warp_loss

parser = argparse.ArgumentParser(description="Self-supervised Multiscale Adversarial Regression Network.")
parser.add_argument('--dataset_root', default='/data/SMAR-Dataset')
parser.add_argument('--dataset_name', default='Flying3D', choices=['Flying3D', 'KITTI2015', 'Beihang'])
parser.add_argument('--model_name', default='SMAR-Net')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--workers', default=1, type=int)
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--log', default='progress_log.csv')
parser.add_argument('--pretrained', default=None)
parser.add_argument('--height', default=512, type=int)
parser.add_argument('--width', default=256, type=int)
parser.add_argument('--disp_max', default=128, type=int)
parser.add_argument('--is_debug', default=True)

args = parser.parse_args()

if args.dataset_name == "Flying3D":
    from data_loader.flying3D_loader import SMARDataLoader
elif args.dataset_name == "KITTI2015":
    from data_loader.kitti_2015_loader import SMARDataLoader
elif args.dataset_name == "Beihang":
    from data_loader.beihang_loader import SMARDataLoader

print("Start Training!")
torch.manual_seed(1024)
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("1. Path to save the output")
save_path = Path(save_path_formatter((args)))
args.save_path = 'checkpoints' / save_path
args.save_path.makedirs_p()
print("=> will save everything to {}".format(args.save_path))

print("2. Data Loading...")

train_transform = custom_transforms.Compose([
    custom_transforms.RandomCrop(args.height, args.width),
    custom_transforms.ArrayToTensor(),
])

valid_transform = custom_transforms.Compose([
    custom_transforms.RandomCrop(args.height, args.width),
    custom_transforms.ArrayToTensor(),
])

train_set = SMARDataLoader(transform=train_transform, train=True)
val_set = SMARDataLoader(transform=valid_transform, train=False)

print('{} samples found in train split'.format(len(train_set.left_img_list)))
print('{} samples found in val split'.format(len(val_set.left_img_list)))

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memort=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

print("3. Creating Model")

model_R = models.SMAR_R(disp_max=args.disp_max).to(device)
model_D = models.SMAR_D().to(device)

if args.pretrained is not None:
    weights = torch.load(args.pretrained)
    model_R_weight = weights['model_R_state_dict']
    model_D_weight = weights['model_D_state_dict']
    model_R.load_state_dict(model_R_weight, strict=False)
    model_D.load_state_dict(model_D_weight, strict=False)

print("4.Setting Optimization Solver")

optimizer_R = torch.optim.Adam(model_R.parameters(), lr=0.001, weight_decay=1e-4)
exp_lr_scheduler_R = lr_scheduler.StepLR(optimizer_R, step_size=250, gamma=0.5)

optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.001, weight_decay=1e-4)
exp_lr_scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=250, gamma=0.5)

print("5.Start TensorboardX")

train_writer = SummaryWriter(args.save_path / 'train')
valid_writer = SummaryWriter(args.save_path / 'valid')

print("6. Create csvfile to save log information.")

with open(args.save_path / args.log, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimeiter='\t')
    csv_writer.writerow(['train_loss', 'val_loss'])

print("7. Start Training!")


def main():
    best_error = -1

    for epoch in range(args.epochs):
        start_time = time.time()
        errors, error_names = valid(model_R, val_loader)
        error_string = ','.join('{} : {".6f}'.format(name, error) for name, error in zip(error_names, errors))

        losses, loss_names = train(model_R, model_D, train_loader, optimizer_R, optimizer_D)
        loss_string = ','.join('{}:{:.6f}'.format(name, loss) for name, loss in zip(loss_names, losses))

        print('Epoch"{},{},{},Time Cost:{}'.format(epoch, error_string, loss_string, time.time() - start_time))

        for loss, name in zip(losses, loss_names):
            train_writer.add_scalar(name, loss, epoch)

        for error, name in zip(errors, error_names):
            valid_writer.add_scalar(name, error, epoch)

        with open(args.save_path / args.log, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow([losses[0], losses[1]])

            decisive_error = errors[0]
            if best_error < 0:
                best_error = decisive_error

            is_best = decisive_error < best_error
            best_error = min(best_error, decisive_error)
            torch.save({
                'epoch': epoch + 1,
                'model_R_state_dict': model_R.state_dict(),
                'model_D_state_dict': model_D.state_dict()
            }, args.save_path / 'last.tar')

            if is_best:
                shutil.copyfile(args.save_path / 'last.tar', args.save_path / 'best_tar')


def train(model_R, model_D, train_loader, optimizer_R, optimizer_D):
    loss_names = ['Train loss', 'Warp loss', 'adversarial loss']
    losses = AverageMeter(i=len(loss_names))

    model_R.train()
    model_D.train()
    exp_lr_scheduler_D.step()
    exp_lr_scheduler_R.step()

    for i, (left_img, right_img) in enumerate(train_loader):

        left_img = left_img.to(device)
        right_img = right_img.to(device)

        label_r, label_f = torch.zeros((1,1)), torch.ones((1,1))

        label_r, label_f = label_r.to(device), label_f.to(device)

        disp_pre = model_R(left_img,right_img)
        img_pre, img = warp_image(disp_pre,left_img,right_img)
        loss = Img_warp_loss(disp_pre,left_img,right_img)
        optimizer_R.zero_grad()

        output_r = model_D(img)

        errD_real = torch.nn.BCELoss(output_r,label_r)
        errD_real.backward()
        optimizer_D.step()

        output_f = model_D(left_img)
        errD_fake = torch.nn.BCELoss(output_f,label_f)
        errD_fake.backward()
        optimizer_D.step()

        optimizer_R.zero_grad()
        output = model_R(disp_pre)
        errG = Img_warp_loss(output, label_r)
        G_loss = loss + errG
        G_loss.backward()
        optimizer_R.step()

        losses.update([G_loss.item(),loss.item(),errG.item()],args.batch_size)

    return losses.avg, loss_names


def warp_image(disp, left_img, right_img):
    B, C, H, W = disp.size()

    # mesh grid
    grid = torch.arange(0, W).view(1, -1).repeat(H, 1)
    grid = grid.view(1, 1, H, W).repeat(B, 1, 1, 1)

    vgrid = grid - disp

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(right_img, vgrid)
    mask = torch.ones(right_img.size())
    mask = torch.nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    img_warp_mask = output * mask
    img_mask = left_img * mask
    img_mask.detach_()

    return img_warp_mask,img_mask

def valid():
    error_names = ['valid loss']
    errors = AverageMeter(i=len(error_names))

    with torch.no_grad():
        for i, (left_img,right_img) in enumerate(val_loader):
            left_img = left_img.to(device)
            right_img = right_img.to(device)
            disp_pre = model_R(left_img, right_img)
            error = Img_warp_loss(disp_pre, left_img, right_img)
            errors.update([error.item()],args.batch_size)

    return errors.avg, error_names

if __name__ == '__main__':
    main()
