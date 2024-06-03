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
# DPT模型加载
pipe = pipeline(task="depth-estimation", model='model_cache/AI-ModelScope/dpt-large')

# 基本的深度神经网络，基于光圈信息对渲染参数进行推理。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SBTNet(nn.Module):
    def __init__(self):
        super(SBTNet, self).__init__()
        self.pipe = pipeline(task="depth-estimation", model='model_cache/AI-ModelScope/dpt-large')
        self.f_convert = torch.load('model_cache/f_alpha/model.bin')
    
    def forward(self, original:Image.Image, src_f, tgt_f):
        # 渲染参数推理
        alpha, depth = self.f_convert(torch.Tensor([src_f, tgt_f]))
        # 大光圈效果渲染
        bokeh = self.create_bokeh_effect(
            original=original, 
            alpha = float(alpha), 
            depth_amount = float(depth),
            circle_shape = (1,1)
        )
        return bokeh
        
    
    def create_bokeh_effect(self, original:Image.Image, alpha:float, depth_amount:float, circle_shape:tuple):
        # Convert the image to RGB
        original = original.convert("RGB")

        # Get the depth map
        result = self.pipe(original)
        depth_map = result["depth"]

        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

        # Expand the depth map to 3 channels
        depth_map = np.repeat(np.expand_dims(depth_map, axis=2), 3, axis=2)

        # Convert the PIL Image to a numpy array
        img = np.array(original)

        # Create the blur image
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=alpha * 10)

        # Adjust the circle shape
        blur = cv2.dilate(blur, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, circle_shape))

        # # Create the bokeh image
        bokeh = np.clip(depth_amount * depth_map * img + (1 - depth_amount * depth_map) * blur, 0, 255).astype(np.uint8)
        # bokeh = blur

        # Adjust the focus depth
        bokeh[depth_map > depth_amount] = img[depth_map > depth_amount]

        # Convert the numpy array back to a PIL Image
        bokeh = Image.fromarray(bokeh)

        return bokeh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--root_folder', required=False, default='TEST_ROOT_FOLDER', type=str)
    parser.add_argument('--save_folder', required=False, default='SAVE_FOLDER', type=str)
    parser.add_argument('--src_f', default=16.0, type=float)
    parser.add_argument('--tgt_f', default=1.8, type=float)
    args = parser.parse_args()
    sbt_model = SBTNet()
    original_list = [f for f in list(os.walk(args.root_folder))[0][2] if f.endswith('.jpg')]
    for original_file in tqdm(original_list):
        original = Image.open(os.path.join(args.root_folder, original_file))
        bokeh = sbt_model(original, args.src_f, args.tgt_f)
        bokeh.save(os.path.join(args.save_folder, original_file))