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


if __name__ == '__main__':
    root_folder = 'TEST_ROOT_FOLDER'
    save_folder = 'SAVE_FOLDER'
    pipe = pipeline(task="depth-estimation", model='model_cache/AI-ModelScope/dpt-large')
    original_list = [f for f in list(os.walk(root_folder))[0][2] if f.endswith('.jpg')]
    for original_file in original_list:
        original = Image.open(os.path.join(root_folder, original_file))
        # 虚拟大光圈渲染
        result = pipe(original)
        depth_map = result["depth"]
        depth_map.save(os.path.join(save_folder, original_file.replace('src', 'alpha').replace('jpg', 'png')))