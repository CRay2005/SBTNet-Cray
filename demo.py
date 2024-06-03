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
from torch import nn
import torch.optim as optim
from model import SBTNet
from PIL import Image

from transformers import pipeline

# 加载DPT模型进行深度估计
pipe = pipeline(task="depth-estimation", model='model_cache/AI-ModelScope/dpt-large')

# 定义一个基础的深度神经网络，用于根据光圈信息推断渲染参数
# 下面的self.f_convert对象的类
# 之所以还需要定义Net，是因为若不引用它，Python仅知道self.f_convert是一个名叫Net类的对象，但是不知道Net类是如何定义的。
# 简单来说，不引用或者定义Net，self.f_convert = torch.load('model_cache/f_alpha/model.bin')可以正常运行，
# 但是运行到alpha, depth = self.f_convert(torch.Tensor([src_f, tgt_f]))这句时会报错
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义全连接层
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # 定义前向传播过程
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SBTNetFake(nn.Module):
    def __init__(self):
        super(SBTNetFake, self).__init__()
        # 加载深度估计模型和转换模型
        self.pipe = pipeline(task="depth-estimation", model='model_cache/AI-ModelScope/dpt-large')
        # f_convert
        # 这个模型是通过f_alpha.csv这个数据集训练的。
        # 作用是将src_f, tgt_f转化成alpha，depth对应的参数。
        # 获取f_alpha.csv的原理很简单，就是随机设定一组src_f, tgt_f并计算SBTNet的虚拟大光圈图。
        # 并计算若干组alpha，depth的手动虚拟大光圈图。
        # 然后比较两个虚拟大光圈图，找到足够相似的那组alpha，depth，就成了标注数据了。
        # 简单来说就是通过算力堆。
        # 有点类似前段时间那个学习游戏“王者荣耀”的那个任务，有点强化学习的意味。只不过这次的任务简单得多。
        self.f_convert = torch.load('model_cache/f_alpha/model.bin')
    
    def forward(self, original:Image.Image, src_f, tgt_f):
        # 渲染参数推理
        alpha, depth = self.f_convert(torch.Tensor([src_f, tgt_f]))
        # if src_f > tgt_f:
        #     alpha, depth = self.f_convert(torch.Tensor([src_f, tgt_f]))
        # else:
        #     alpha, depth = self.f_convert(torch.Tensor([tgt_f, src_f]))
        #     alpha = -alpha
        # 大光圈效果渲染
        bokeh, depth_map = self.create_bokeh_effect(
            original=original, 
            alpha = float(alpha), 
            depth_amount = float(depth),
            circle_shape = (1,1)
        )
        return bokeh, depth_map
        
    
    def create_bokeh_effect(self, original:Image.Image, alpha:float, depth_amount:float, circle_shape:tuple):
        # Convert the image to RGB
        original = original.convert("RGB")

        # Get the depth map
        # 深度信息渲染
        result = self.pipe(original)
        depth_map = result["depth"]

        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

        # Expand the depth map to 3 channels
        _depth_map = np.repeat(np.expand_dims(depth_map, axis=2), 3, axis=2)

        # Convert the PIL Image to a numpy array
        img = np.array(original)

        # print(img)
        # Create the blur image
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=alpha * 10)
        # if alpha > 0:
        #     blur = cv2.GaussianBlur(img, (0, 0), sigmaX=alpha * 10)
        # else:
        #     blur = cv2.GaussianBlur(img, (0, 0), sigmaX=-alpha * 10)
        #     print(blur)
        #     blur = cv2.subtract(img, blur)
        #     print(blur)
        # Adjust the circle shape
        blur = cv2.dilate(blur, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, circle_shape))

        # # Create the bokeh image
        bokeh = np.clip(depth_amount * _depth_map * img + (1 - depth_amount * _depth_map) * blur, 0, 255).astype(np.uint8)
        # bokeh = blur

        # Adjust the focus depth
        bokeh[_depth_map > depth_amount] = img[_depth_map > depth_amount]

        # Convert the numpy array back to a PIL Image
        bokeh = Image.fromarray(bokeh)

        return bokeh, depth_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--root_folder', required=False, default='TEST_ROOT_FOLDER', type=str)
    parser.add_argument('--save_folder', required=False, default='SAVE_FOLDER', type=str)
    parser.add_argument('--src_f', default=16.0, type=float)
    parser.add_argument('--tgt_f', default=1.8, type=float)
    args = parser.parse_args()
    # 创建模型
    sbt_model = SBTNetFake()
    # 读取文件名
    original_list = [f for f in list(os.walk(args.root_folder))[0][2] if f.endswith('.jpg')]
    for original_file in tqdm(original_list):
        # 读取照片
        original = Image.open(os.path.join(args.root_folder, original_file))
        # 虚拟大光圈渲染
        bokeh, depth_map = sbt_model(original, args.src_f, args.tgt_f)
        # 保存
        bokeh.save(os.path.join(args.save_folder, original_file))
        # 这里没有输出pred_alpha, 再train.py中才有相关内容
        # depth_map = Image.fromarray(depth_map)
        # depth_map.save(os.path.join(args.save_folder, original_file.replace('src', 'alpha').replace('jpg', 'png')))