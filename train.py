#!/usr/bin/env python
# encoding: utf-8

import os
import time
import glob
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse

import torch
import torch.optim as optim
from PIL import Image

from model import SBTNet
# from evaluation import read_meta_data
# Import Net class is necessary
from demo import SBTNetFake, Net


def read_meta_data(meta_file_path):
    if not os.path.isfile(meta_file_path):
        raise ValueError(f"Meta file missing under {meta_file_path}.")
    meta = {}
    with open(meta_file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        id, src_lens, tgt_lens, disparity = [part.strip() for part in line.split(",")]
        meta[id] = (src_lens, tgt_lens, disparity)
    return meta


# 加载预训练模型
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--root_folder', required=False, default='TEST_ROOT_FOLDER', type=str)
parser.add_argument('--save_folder', required=False, default='SAVE_FOLDER', type=str)
parser.add_argument('--pretrained_path', type=str, default='checkpoints/model.pth')
parser.add_argument('--gpus', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

src_paths = sorted(glob.glob(os.path.join(args.root_folder, "*.src.jpg")))
meta_data = read_meta_data(os.path.join(args.root_folder, "meta.txt"))

os.makedirs(args.save_folder, exist_ok=True)
#Cray
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
#model = SBTNet().cuda()
model = SBTNet().to(device)

model = torch.nn.DataParallel(model)

checkpoint = torch.load(args.pretrained_path, map_location='cpu')

model.load_state_dict(checkpoint, strict=True)

# 如果有GPU可用，将模型移动到GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 建立模型
model = model.module
model = model.to(device)

# 加载仿真模型，用于生成虚拟大光圈效果图。
model_fake = SBTNetFake()


# 定义损失函数和优化器
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 利用DPT仿真标注数据
# from transformers import pipeline
# pipe = pipeline(task="depth-estimation", model='model_cache/AI-ModelScope/dpt-large')

# 开始训练循环
torch.cuda.empty_cache()
for epoch in tqdm(range(5000), desc='training'):  
    running_loss = 0.0
    torch.cuda.empty_cache()
    # for i in range(len(src_paths)):
    # 论文里，该模型是在4路GPU上训练的。在这里仅训练一个样本证明猜想
    for i in [0]:
        src_path = src_paths[i]
        if os.path.isdir(src_path):
            continue
        # 读数据
        src = cv2.imread(src_path).astype(np.float32)[..., ::-1] / 255
        src = torch.from_numpy(src).permute(2, 0, 1)

        # 回归原始逻辑
        filename = os.path.basename(src_path)
        id = filename.split(".")[0]
        src_lens, tgt_lens, disparity = meta_data[id]
        src_lens_type = torch.tensor(len(src_lens.split('50mmf')[0]) - 4, dtype=torch.float32)  # Sony: 0, Canon: 1
        tgt_lens_type = torch.tensor(len(tgt_lens.split('50mmf')[0]) - 4, dtype=torch.float32)  # Sony: 0, Canon: 1
        src_F = torch.tensor(float(src_lens.split('50mmf')[1][:-2]), dtype=torch.float32)
        tgt_F = torch.tensor(float(tgt_lens.split('50mmf')[1][:-2]), dtype=torch.float32)
        disparity = torch.tensor(float(disparity), dtype=torch.float32) / 100

        h, w = src.shape[1:]
        cateye_x, cateye_y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-h / w, h / w, h))
        cateye_coord = np.concatenate([cateye_x[..., None], cateye_y[..., None]], axis=-1).astype(np.float32)
        cateye_coord = torch.from_numpy(cateye_coord).permute(2, 0, 1)
        
        # 将数据灌入GPU，为后面的计算做准备
        src = src[None].to(device)
        src_lens_type = src_lens_type[None].to(device)
        tgt_lens_type = tgt_lens_type[None].to(device)
        src_F = src_F[None].to(device)
        tgt_F = tgt_F[None].to(device)
        disparity = disparity[None].to(device)
        cateye_coord = cateye_coord[None].to(device)

        # torch.cuda.synchronize()
        t0 = time.time()

        # 零参数梯度
        optimizer.zero_grad()

        # 前向+后向+优化
        # outputs = model(inputs)
                ##### disable AlphaNet while testing real-world images #####
        # print(src.shape)
        # 前向推理
        if 'real_' not in src_path:
            pred, pred_alpha = model(src, src_lens_type, tgt_lens_type, src_F, tgt_F, disparity, cateye_coord, use_alpha=True)
        else:
            pred, pred_alpha = model(src, src_lens_type, tgt_lens_type, src_F, tgt_F, disparity, cateye_coord, use_alpha=False)
        ############################################################
        # print(pred_alpha.shape)
        # 计算标注数据（虚拟大光圈渲染图）
        original = Image.open(src_path)
        tgt, tgt_alpha = model_fake(original, src_F, tgt_F)
        tgt_array = np.array(tgt)
        tgt_array = tgt_array.astype(np.float32)[..., ::-1]/255
        tgt_alpha_array = np.array(tgt_alpha)
        tgt_alpha_array = tgt_alpha_array.astype(np.float32)
        tgt_alpha_array = tgt_alpha_array.reshape([1]+list(tgt_alpha_array.shape))
        # print(tgt_alpha_array.shape, pred_alpha.shape)
        # print(tgt_alpha_array)
        # print(pred_alpha)
        # tgt = torch.unsqueeze(torch.from_numpy(tgt_array).permute(2,0,1), dim=0)
        tgt_array = torch.from_numpy(tgt_array).permute(2,0,1)
        tgt_array = tgt_array[None].to(device)
        tgt_alpha_array = torch.from_numpy(tgt_alpha_array)
        tgt_alpha_array = tgt_alpha_array[None].to(device)
        
        # 分别计算深度图的MSE损失函数与目标结果的MSE损失函数
        print(tgt_alpha_array.shape, pred_alpha.shape)
        print(pred.shape, tgt_array.shape)
        loss_alpha = criterion(pred_alpha, tgt_alpha_array)
        loss_array = criterion(pred, tgt_array)
        # 这里的损失函数是为了三原色三个通道分别计算损失函数
        # 原因是，上述两个损失函数训练完，还会存在偏色问题
        # 所以，试图分别计算三个颜色的损失函数，然后让标准差最小化，来避免偏色。
        # loss_red = criterion(pred[:,0,:,:], tgt_array[:,0,:,:])
        # loss_green = criterion(pred[:,1,:,:], tgt_array[:,1,:,:])
        # loss_blue = criterion(pred[:,2,:,:], tgt_array[:,2,:,:])
        if epoch > 1000:
            loss = loss_alpha + loss_array
            # loss = loss_alpha + loss_array + torch.std(torch.tensor([loss_red, loss_green, loss_blue]))
        else:
            loss = loss_alpha
        # print(float(loss_alpha), float(loss_array), float(loss), float(loss_red), float(loss_green), float(loss_blue))
        # print(torch.std(torch.tensor([loss_red, loss_green, loss_blue])), torch.mean(torch.tensor([loss_red, loss_green, loss_blue])))
        
        # 计算反馈（本质就是梯度）
        # 所以神经网络也称作前馈神经网络，一前，一馈，完整一轮训练流程。
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        # if i % 2 == 1999:    # 每2000个小批量打印一次
        #     print('[%d, %5d] loss: %.3f' %
        #             (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0
        
        # 保存虚拟大光圈渲染图像
        pred = pred.clamp(0, 1)
        # torch.cuda.synchronize()

        pred = pred[0].permute(1, 2, 0).detach().cpu().numpy()

        save_path = os.path.join(args.save_folder, os.path.basename(src_path).replace('.src', '.src_pred'))
        cv2.imwrite(save_path, pred[..., ::-1] * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        # 保存深度图
        pred_alpha = pred_alpha.clamp(0, 1)
        # torch.cuda.synchronize()

        pred_alpha = pred_alpha[0].permute(1, 2, 0).detach().cpu().numpy()

        save_path = os.path.join(args.save_folder, os.path.basename(src_path).replace('.src', '.src_alpha'))
        # print(save_path)
        cv2.imwrite(save_path, pred_alpha[..., ::-1] * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        
        tgt_alpha_array = tgt_alpha_array.clamp(0, 1)
        # torch.cuda.synchronize()

        tgt_alpha_array = tgt_alpha_array[0].permute(1, 2, 0).detach().cpu().numpy()

        save_path = os.path.join(args.save_folder, os.path.basename(src_path).replace('.src', '.src_label'))
        # print(save_path)
        cv2.imwrite(save_path, tgt_alpha_array[..., ::-1] * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


    
print('Finished Training')
