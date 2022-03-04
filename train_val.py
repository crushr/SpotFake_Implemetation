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


def train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, device='cpu', 
            param_dict_model=None, param_dict_opt=None, save_best=False, file_path='/home/madm/Documents/multi_model/saved_models/best_model.pt',
            writer=None
            ):
    
    # 开始 training_loop
    best_acc_val = 0
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # 打印标题

        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*100)

        # 记录时间
        t0_epoch, t0_batch = time.time(), time.time()

        # 重置损失
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # 训练模式
        model.train()

        # batch训练
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1

            img_ip , text_ip, label = batch["image_id"], batch["BERT_ip"], batch['label']
            
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)
            
            imgs_ip = img_ip.to(device)
            
            b_labels = label.to(device)

            # 梯度清零
            model.zero_grad()

            # 向前传播
            # logits, att_mask_img = model(text=[b_input_ids, b_attn_mask], image=imgs_ip, label=b_labels)
            logits = model(text=[b_input_ids, b_attn_mask], image=imgs_ip)

            # 计算、累计损失
            b_labels=b_labels.to(torch.float32)
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数和学习率
            optimizer.step()
            scheduler.step()


            # 每20个batch打印损失和时间
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                # 打印训练情况
                print(f"epoch{epoch_i + 1:^7} | batch{step:^7} | loss{batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | elapsed{time_elapsed:^9.2f}")
                
                # 写入tensorboard
                if writer != None:
                    writer.add_scalar('Training Loss', (batch_loss / batch_counts), epoch_i*len(train_dataloader)+step)
                
                # 重置
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # 计算整个训练集的平均损失
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*100)

        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # 在每个epoch训练完成后，测试模型的性能
            val_loss, val_accuracy = evaluate(model, loss_fn, val_dataloader, device)
            
            time_elapsed = time.time() - t0_epoch
            print(f" {epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
            
            # 写入tensorboard
            if writer != None:
                writer.add_scalar('Validation Loss', val_loss, epoch_i+1)
                writer.add_scalar('Validation Accuracy', val_accuracy, epoch_i+1)
            
            # 保存best_model
            if save_best: 
                if val_accuracy > best_acc_val:
                    best_acc_val = val_accuracy
                    torch.save({
                                'epoch': epoch_i+1,
                                'model_params': param_dict_model,
                                'opt_params': param_dict_opt,
                                'model_state_dict': model.state_dict(),
                                'opt_state_dict': optimizer.state_dict(),
                                'sch_state_dict': scheduler.state_dict()
                               }, file_path)
                    
        print("\n")
    
    print("Training complete!")
    
    
    
def evaluate(model, loss_fn, val_dataloader, device):
    """
        在每个epoch训练完成后，测试模型的性能
    """

    model.eval()

    # 记录损失和准确率
    val_accuracy = []
    val_loss = []

    # 每个损失之后
    for batch in val_dataloader:
        img_ip , text_ip, label = batch["image_id"], batch["BERT_ip"], batch['label']
            
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)

        imgs_ip = img_ip.to(device)

        b_labels = label.to(device)

        # 计算logits
        with torch.no_grad():
            # logits, att_mask_img = model(text=[b_input_ids, b_attn_mask], image=imgs_ip, label=b_labels)
            logits = model(text=[b_input_ids, b_attn_mask], image=imgs_ip)
            b_labels=b_labels.to(torch.float32)
            
        # 计算loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        logits[logits<0.5] = 0
        logits[logits>=0.5] = 1
        # print(logits)
        # 预测 
        # preds = torch.argmax(logits, dim=1).flatten()
        #print(preds)

        # 计算准确率
        accuracy = (logits == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # 计算平均准确率和验证集损失
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy