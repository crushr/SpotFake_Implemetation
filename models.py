from importlib_metadata import re
import torch
import numpy as np
import transformers
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from transformers import BertModel

device = torch.device("cuda")

# 文本Bert基本模型
class TextEncoder(nn.Module):

    def __init__(self, text_fc2_out=32, text_fc1_out=2742, dropout_p=0.4, fine_tune_module=False):

        super(TextEncoder, self).__init__()
        
        self.fine_tune_module = fine_tune_module

        # 实例化
        self.bert = BertModel.from_pretrained(
                    'bert-base-uncased',
#                     output_attentions = True, 
                    return_dict=True)

        self.text_enc_fc1 = torch.nn.Linear(768, text_fc1_out)

        self.text_enc_fc2 = torch.nn.Linear(text_fc1_out, text_fc2_out)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()
        
    def forward(self, input_ids, attention_mask):
        """
        输入Bert和分类器，计算logis

        @参数    input_ids (torch.Tensor): 输入 (batch_size,max_length)

        @参数    attention_mask (torch.Tensor): attention mask information (batch_size, max_length)

        @返回    logits (torch.Tensor): 输出 (batch_size, num_labels)
        """

        # 输入BERT
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # print(out['pooler_output'].shape)
        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc1(out['pooler_output']))
        )    
        
        x = self.dropout(
            torch.nn.functional.relu(
                self.text_enc_fc2(x))
        ) 
        
        return x
    
    def fine_tune(self):
        """
        固定参数
        """
        for p in self.bert.parameters():
            p.requires_grad = self.fine_tune_module
            

# 视觉vgg19预训练模型
class VisionEncoder(nn.Module):
   
    def __init__(self, img_fc1_out=2742, img_fc2_out=32, dropout_p=0.4, fine_tune_module=False):
        super(VisionEncoder, self).__init__()
        
        self.fine_tune_module = fine_tune_module
        
        # 实例化
        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        
        self.vis_encoder = vgg

        self.vis_enc_fc1 = torch.nn.Linear(4096, img_fc1_out)

        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)

        self.dropout = nn.Dropout(dropout_p)

        self.fine_tune()
        
    def forward(self, images):
        """
        :参数: images, tensor (batch_size, 3, image_size, image_size)
        :返回: encoded images
        """

        x = self.vis_encoder(images)

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )

        return x
    
    def fine_tune(self):
        """
        允许或阻止vgg的卷积块2到4的梯度计算。
        """
        for p in self.vis_encoder.parameters():
            p.requires_grad = False

        # 如果进行微调，则只微调卷积块2到4
        for c in list(self.vis_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_module

#LanguageAndVisionConcat
class Text_Concat_Vision(torch.nn.Module):

    def __init__(self,
        model_params
    ):
        super(Text_Concat_Vision, self).__init__()
        
        self.text_encoder = TextEncoder(model_params['text_fc2_out'], model_params['text_fc1_out'], model_params['dropout_p'], model_params['fine_tune_text_module'])
        self.vision_encode = VisionEncoder(model_params['img_fc1_out'], model_params['img_fc2_out'], model_params['dropout_p'], model_params['fine_tune_vis_module'])

        self.fusion = torch.nn.Linear(
            in_features=(model_params['text_fc2_out'] + model_params['img_fc2_out']), 
            out_features=model_params['fusion_output_size']
        )
        self.fc = torch.nn.Linear(
            in_features=model_params['fusion_output_size'], 
            out_features=1
        )
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])


    #def forward(self, text, image, label=None):
    def forward(self, text, image, label=None):

        ## text to Bert
        text_features = self.text_encoder(text[0], text[1])
        ## image to vgg
        image_features = self.vision_encode(image)

        ## 连接image & text 
        combined_features = torch.cat(
            [text_features, image_features], dim = 1
        )

        combined_features = self.dropout(combined_features)
        
        fused = self.dropout(
            torch.relu(
            self.fusion(combined_features)
            )
        )
        
        # prediction = torch.nn.functional.sigmoid(self.fc(fused))
        prediction = torch.sigmoid(self.fc(fused))

        prediction = prediction.squeeze(-1)

        # prediction = prediction.cpu().detach().numpy()
        
        # for i in range(len(prediction)):
        #     if prediction[i] > 0.5:
        #         prediction[i] = 1.0
        #     else:
        #         prediction[i] = 0.0

        # prediction = torch.tensor(prediction, dtype=torch.float32, requires_grad = False).to(device)
        prediction = prediction.float()

        return prediction