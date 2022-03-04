import torch
import pandas as pd
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import os
import re
import math

from torch.utils.tensorboard import SummaryWriter

from models import *
from data_loader import *
from train_val import *


df_train = pd.read_csv("/home/madm/Documents/multi_model/twitter/train_posts_clean.csv")
df_test = pd.read_csv("/home/madm/Documents/multi_model/twitter/test_posts.csv")

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
# 图像转换
image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# 实例化 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# 固定最大长度
MAX_LEN = 500
root_dir = "/home/madm/Documents/multi_model/twitter/"

# 读取数据
transformed_dataset_train = FakeNewsDataset(df_train, root_dir+"images_train/", image_transform, tokenizer, MAX_LEN)

transformed_dataset_val = FakeNewsDataset(df_test, root_dir+"images_test/", image_transform, tokenizer, MAX_LEN)

train_dataloader = DataLoader(transformed_dataset_train, batch_size=8,
                        shuffle=True, num_workers=0)

val_dataloader = DataLoader(transformed_dataset_val, batch_size=8,
                        shuffle=True, num_workers=0)


# 损失
loss_fn = nn.BCELoss()

def set_seed(seed_value=42):
    """
        设置种子
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

parameter_dict_model={
    'text_fc2_out': 32, 
    'text_fc1_out': 2742, 
    'dropout_p': 0.4, 
    'fine_tune_text_module': False,
    'img_fc1_out': 2742, 
    'img_fc2_out': 32, 
    'dropout_p': 0.4, 
    'fine_tune_vis_module': False,
    'fusion_output_size': 35
}

parameter_dict_opt={'l_r': 3e-5,
                    'eps': 1e-8
                    }


EPOCHS=50

# 设置随机种子
set_seed(7)

final_model = Text_Concat_Vision(parameter_dict_model)

final_model = final_model.to(device) 

# 优化器
optimizer = AdamW(final_model.parameters(),
                  lr=parameter_dict_opt['l_r'],
                  eps=parameter_dict_opt['eps'])

# training steps总数
total_steps = len(train_dataloader) * EPOCHS

# 学习率衰减
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # 默认值
                                            num_training_steps=total_steps)

# 启动tensorboard
writer = SummaryWriter('/home/madm/Documents/multi_model/runs/multi_att_exp3')

# 开始
train(model=final_model,
      loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
      train_dataloader=train_dataloader, val_dataloader=val_dataloader,
      epochs=150, evaluation=True,
      device=device,
      param_dict_model=parameter_dict_model, param_dict_opt=parameter_dict_opt,
      save_best=True,
      file_path='/home/madm/Documents/multi_model/saved_models/best_model.pt'
      , writer=writer
      )