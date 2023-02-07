import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('.\.')

import cv2
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models import resnet50
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from loss import Matrix_Loss

class FCN(nn.Module):
    def __init__(self, num_classes=1):
        super(FCN, self).__init__()
        backbone = resnet50(pretrained=True)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=1)
        )
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        )
        
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        T5 = self.relu(self.deconv1(x4)) # size=(N, 512, x.H/16, x.W/16)
        T5 = self.bn1(T5 + x3) # 特征逐点相加
        T4 = self.relu(self.deconv2(T5)) 
        T4 = self.bn2(T4 + x2)
        T3 = self.bn3(self.relu(self.deconv3(T4))) # size=(N, 128, x.H/4, x.W/4)
        T2 = self.bn4(self.relu(self.deconv4(T3))) # size=(N, 64, x.H/2, x.W/2)
        T1 = self.bn5(self.relu(self.deconv5(T2))) # size=(N, 32, x.H, x.W)
        out = self.classifier(T1)

        return out
    
class EgoAvoidance(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        sub_files = os.listdir(self.root)
        self.data = []
        for sub_file in sub_files:
            sub_file_path = os.path.join(self.root, sub_file)
            json_dir = os.path.join(sub_file_path, 'traj_jsons')
            json_names = os.listdir(json_dir)
            for json_name in json_names:
                json_fn = os.path.join(json_dir, json_name)
                im_fn = os.path.join(sub_file_path, 'im', json_name.replace('.json', '.png'))
                if os.path.exists(im_fn):
                    with open(json_fn, 'r') as f:
                        data = json.load(f)
                    if 'traj_iuvs' in data and (len(data['traj_iuvs'])>0):
                        self.data.append({'im_path':im_fn, 'traj_iuvs':data['traj_iuvs']})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        info = self.data[idx]
        im_fn = info['im_path']
        ego_kps = np.array(info['traj_iuvs'])

        ego_mask = (ego_kps[:, 0] > 0) & (ego_kps[:, 1] > 0) 
        ego_kps = ego_kps[ego_mask].astype(np.int0)

        rgb = Image.open(im_fn).convert('RGB')

        mask = np.zeros((rgb.height, rgb.width)).astype(np.float32)
        for idx, kp in enumerate(ego_kps):
            curr_point = kp
            if idx == 0:
                last_point = curr_point
                continue
            mask = cv2.line(mask, last_point, curr_point, color = 1, thickness=30)
            last_point = curr_point
        rgb = self.transform(rgb)
        mask = cv2.resize(mask, dsize=(rgb.shape[1], rgb.shape[2]))
        mask = 1 - cv2.GaussianBlur(mask, ksize=(55, 55), sigmaX=0)

        mask = torch.from_numpy(mask)
        
        return rgb, mask

# 学习率的设置
def adjust_learning_rate(optimizer, i_iter, Max_step):
    """Sets the learning rate to the initial LR divided"""

    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    lr = lr_poly(0.01, i_iter, Max_step, 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # optimizer.param_groups[0]['lr'] = lr
    return lr

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('--root', default='/mnt/lq/uisee/dogdog/Dataset',help='test config file path')
    parser.add_argument('--batch_size', type=int, default=8, help='CUDA device id')
    parser.add_argument('--num_workers', type=int, default=2, help='CUDA device id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    normalize = transforms.Normalize(mean=[0.416, 0.42, 0.412],
                                 std=[0.204, 0.2, 0.209])
    ego_avoid_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        normalize,]
    )
    
    dataset = EgoAvoidance(args.root, transform=ego_avoid_transforms)
    TrainDataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = FCN(num_classes=1)

    model = model.cuda()
    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 0.01}],
        lr=0.01, momentum=0.9, weight_decay=0.0001)

    optimizer.zero_grad()

    loss_func = Matrix_Loss(kld_parm=1)
    epoch = 24
    for epoch_i in range(epoch):
        lr = adjust_learning_rate(optimizer, epoch_i, epoch)
        loop = tqdm(enumerate(TrainDataLoader), total=len(TrainDataLoader))
        for index, (rgb, mask) in enumerate(tqdm(TrainDataLoader)):
            rgb = rgb.cuda()
            mask = mask.cuda()
            pred = model(rgb)
            pred = torch.sigmoid(pred)
            loss, kld, cc, sim = loss_func(pred, mask.to(pred.dtype))
     
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            #更新信息
            loop.set_description(f'Epoch [{epoch}/{epoch_i}]')
            loop.set_postfix(lr = lr, loss = loss.item(), kld=kld.item())
        torch.save(model.state_dict(), 'checkpoint/%d.pth'%(epoch_i+1))


