import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import cv2
import PIL
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm

import torch
import torch.nn as nn

VALUE_MAX = 0.05
VALUE_MIN = 0.01
from g1fitting import build_clothoid, points_on_clothoid
from math import pi

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


if __name__ == "__main__":

    model =FCN(num_classes=1)
    checkpoint = torch.load('checkpoint/24.pth')
    model.load_state_dict(checkpoint)
    model = model.cuda().eval()

    normalize = transforms.Normalize(mean=[0.416, 0.42, 0.412],
                                 std=[0.204, 0.2, 0.209])

    ego_avoid_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        normalize,]
    )

    im_dir = 'Test_data/20150418_costco/im'
    ims = os.listdir(im_dir)
    for im_name in ims:
        im_fn = os.path.join(im_dir, im_name)

        im = cv2.imread(im_fn)
        im_ori = cv2.resize(im, (500, 500))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = PIL.Image.fromarray(im)

        rgb = ego_avoid_transforms(im)

        with torch.no_grad():
            pred = model(rgb.cuda().unsqueeze(0))
            affordance = torch.sigmoid(pred)
            affordance = F.interpolate(
                affordance,
                size=(500, 500),
                mode='bilinear',
                align_corners=False,
            )
            affordance = (affordance - affordance.min()) / (affordance.max() - affordance.min())
            pred = affordance.squeeze().squeeze().cpu().numpy()* 255

            pred = cv2.applyColorMap(pred.astype(np.uint8), cv2.COLORMAP_JET)
            im_ori = im_ori * 0.6 + pred * 0.4
            
        # get retrival x_d
        points = []
        retrieval_X = []
        mask = torch.ones(100).to(torch.bool)

        device = affordance.device

        pred = affordance.squeeze().squeeze().cpu().numpy()
        affordance_Guassian = cv2.GaussianBlur(pred,(5,5),0)
        affordance = torch.from_numpy(affordance_Guassian).unsqueeze(dim=0).unsqueeze(dim=0).to(device)

        # 找出起始点
        key_json = os.path.join('Test_data/20150418_costco/traj_jsons/'+im_name[:-4] + ".json")
        if os.path.exists(key_json):
            with open(key_json, 'r') as f:
                data = json.load(f)
                if 'traj_iuvs' not in data:
                    continue
                if len(data['traj_iuvs']) == 0:
                    continue
                start_p = data['traj_iuvs'][0]
                start_p[0] = start_p[0] / 1280 * 500
                start_p[1] = start_p[1] / 960 * 500
        else:
            continue

        """sample"""
        x0 = -(start_p[1] / 499) * 2 +1 
        y0 = (start_p[0] / 499) * 2 - 1 
        theta0 = 0.

        k = 1.0
        dk = 0.0
        L = 2.0
        sample_Xs = []
        cost_min = 100000
        cost_idx = 1000
        idx_sample = 0

        for scale in range(5):
            if scale == 0:
                (y_p, x_p, res2) = points_on_clothoid(x0, y0, theta0, 0, dk, L, 100)
            else:
                k = k*scale*0.4
                (y_p, x_p, res2) = points_on_clothoid(x0, y0, theta0, k, dk, L, 100)

            sample_X = np.hstack([np.array(x_p)[..., None], -np.array(y_p)[..., None]])
            sample_X_mask = (sample_X[:, 0] < 1) & (sample_X[:, 1] < 1) & \
                    (sample_X[:, 0] > - 1) & (sample_X[:, 1] > -1)
            sample_X = sample_X[sample_X_mask]
            sample_X_half_down = sample_X[:sample_X.shape[0] // 2, :]
            if sample_X_half_down.shape[0] == 0:
                continue
            y1, x1 = sample_X_half_down[-1]
            k_up = 1.0

            for scale_up in range(5):
                if scale_up == 0:
                    (y_p_up, x_p_up, res2_up) = points_on_clothoid(x1, y1, theta0, 0, dk, L, 100)
                else:
                    k_up = k_up*scale_up*0.4
                    (y_p_up, x_p_up, res2_up) = points_on_clothoid(x1, y1, theta0, k_up, dk, L, 100)

                sample_X_up = np.hstack([np.array(x_p_up)[..., None], -np.array(y_p_up)[..., None]])
                sample_X_up_mask = (sample_X_up[:, 0] < 1) & (sample_X_up[:, 1] < x1) & \
                    (sample_X_up[:, 0] > - 1) & (sample_X_up[:, 1] > -1)
                sample_X_up = sample_X_up[sample_X_up_mask]
                sample_X = np.vstack([sample_X_half_down[:-1], sample_X_up])
                sample_Xs.append(sample_X)
                
                if scale_up == 0:
                    continue
                sample_X_up = np.hstack([2*y1 - np.array(x_p_up)[..., None], -np.array(y_p_up)[..., None]])
                sample_X_up_mask = (sample_X_up[:, 0] < 1) & (sample_X_up[:, 1] < x1) & \
                    (sample_X_up[:, 0] > - 1) & (sample_X_up[:, 1] > -1)
                sample_X_up = sample_X_up[sample_X_up_mask]
                sample_X = np.vstack([sample_X_half_down[:-1], sample_X_up])
                sample_Xs.append(sample_X)
            if scale == 0:
                continue
            
            sample_X = np.hstack([2*y0 - np.array(x_p)[..., None], -np.array(y_p)[..., None]])
            sample_X_mask = (sample_X[:, 0] < 1) & (sample_X[:, 1] < 1) & \
                    (sample_X[:, 0] > - 1) & (sample_X[:, 1] > -1)
            sample_X = sample_X[sample_X_mask]
            sample_X_half_down = sample_X[:sample_X.shape[0] // 2, :]

            y1, x1 = sample_X_half_down[-1]
            k_up = 1.0

            for scale_up in range(5):
                if scale_up == 0:
                    (y_p_up, x_p_up, res2_up) = points_on_clothoid(x1, y1, theta0, 0, dk, L, 100)
                else:
                    k_up = k_up*scale_up*0.4
                    (y_p_up, x_p_up, res2_up) = points_on_clothoid(x1, y1, theta0, k_up, dk, L, 100)

                sample_X_up = np.hstack([np.array(x_p_up)[..., None], -np.array(y_p_up)[..., None]])
                sample_X_up_mask = (sample_X_up[:, 0] < 1) & (sample_X_up[:, 1] < x1)& \
                    (sample_X_up[:, 0] > - 1) & (sample_X_up[:, 1] > -1)
                sample_X_up = sample_X_up[sample_X_up_mask]
                sample_X = np.vstack([sample_X_half_down[:-1], sample_X_up])
                sample_Xs.append(sample_X)
                
                if scale_up == 0:
                    continue
                sample_X_up = np.hstack([2*y1 - np.array(x_p_up)[..., None], -np.array(y_p_up)[..., None]])
                sample_X_up_mask = (sample_X_up[:, 0] < 1) & (sample_X_up[:, 1] < x1) & \
                    (sample_X_up[:, 0] > - 1) & (sample_X_up[:, 1] > -1)
                sample_X_up = sample_X_up[sample_X_up_mask]
                sample_X = np.vstack([sample_X_half_down[:-1], sample_X_up])
                sample_Xs.append(sample_X)

        for sample_X in sample_Xs:
            sample_x_cuda_grid = torch.from_numpy(sample_X).to(affordance.device)
            if sample_x_cuda_grid.shape[0] >= 100:
                sample_x_cuda_grid = sample_x_cuda_grid[:100]
            else:
                sample_x_cuda_grid_pad = sample_x_cuda_grid.new_zeros([100, 2])
                sample_x_cuda_grid_pad[:sample_x_cuda_grid.shape[0]] = sample_x_cuda_grid
                sample_x_cuda_grid = sample_x_cuda_grid_pad
            #sample_xd = (retrieval_X/499.0)*2.0-1
            batch_size = affordance.shape[0]
            grid = sample_x_cuda_grid.view(batch_size, 1, sample_x_cuda_grid.shape[0], 2).to(torch.float32)
            
            heatmap = F.grid_sample(affordance, grid,
                align_corners=True).permute(0, 2, 1, 3)
            # orientation

            cost = heatmap.sum()
            
            if cost < cost_min:
                cost_min = cost
                cost_idx = idx_sample

            idx_sample += 1
        if cost_idx == 1000:
            continue
        sample_X = sample_Xs[cost_idx]
        sample_X = (((sample_X + 1.0) / 2.0)*499).astype(np.int0)
        for kp_idx, kp in enumerate(sample_X):
            if (kp[0] < 0) or (kp[1] < 0):
               continue
            curr_point = np.array([kp[0], kp[1]])
            if kp_idx == 0:
                last_point = curr_point
                continue
            dist = np.linalg.norm(last_point - curr_point)

            im_ori = cv2.arrowedLine(im_ori, last_point, curr_point, color=(0, 255, 0), thickness=3)
            last_point = curr_point
        save_path = os.path.join('all_case_results/result_sample', im_name+'.png')
        cv2.imwrite(save_path, im_ori)




