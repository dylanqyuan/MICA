import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
import requests
import copy
import torch

from modeling_internvl_chat import InternVLChatModel


def build_visual_backbone():
    # model_id = '/home/ippl/Downloads/data1_2TB/weights/Florence2_large_ft/Florence-2-large-ft'
    model_id = "/home/ippl/Downloads/data1_2TB/weights/mica_250305_cadbcaptionsV21_kupcp_multiTurn_singleTurn.final/weights"

    
    model = InternVLChatModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()  
    return model


class MICAScoring(nn.Module):
    def __init__(self):
        super(MICAScoring, self).__init__()



        self.backbone = build_visual_backbone()
        for param in self.backbone.parameters():
            param.requires_grad = False
        


        # output_channel = input_channel
        self.clca = nn.Sequential(
            nn.Linear(896, 512, bias=False),  # 保持维度不变
            nn.ReLU(),            # 激活函数
            nn.Linear(512, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 896, bias=False),   # 保持维度不变
        ).to(torch.bfloat16)

        
        self.pred_head = nn.Sequential(
            nn.Linear(896, 1024, bias=False),  # 将896维特征映射到512维
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),    # 添加dropout防止过拟合
            nn.Linear(1024, 256, bias=False),  # 进一步降维到256
            nn.ReLU(inplace=True),
            nn.Linear(256, 5, bias=False),    # 最终输出5个分数
            nn.Softmax(dim=1)
        ).to(torch.bfloat16)

               
    def forward(self, x):
        
        # vision related ==============================================
        if x.shape[1] == 1 and x.shape[2] ==3 and len(x.shape)==5:
            x = x.squeeze()

        if x.shape[0] == 3 and x.shape[1] == 224 and x.shape[2] == 224:
            x = x.unsqueeze(0)
        visual_features = self.backbone.extract_feature(x) # [batch, 64, 896]

        # [batch, 64, 896] --> [batch, 896] through torch.mean
        visual_features = torch.mean(visual_features, dim=1, keepdim=False)

        # [batch, 577+1, model_max_length]
        feat_visual_composition = self.clca(visual_features)

        # feat_visual_final = torch.mean(feat_visual_step2, dim=1, keepdim=False)

        dist_pred = self.pred_head(feat_visual_composition)     
        # dist_pred = torch.softmax(dist_pred, dim=-1)  # 对每个batch的5个分数进行softmax归一化
        return dist_pred

    def save_model(self, path):
        """
        Save the model's state dictionary, excluding the backbone parameters.
        
        Args:
            path (str): The file path where the model will be saved.
        """
        # Create a state dictionary excluding the backbone parameters
        state_dict = {
            key: value for key, value in self.state_dict().items()
            if not key.startswith("backbone.")
        }

        # Save the state dictionary to the specified path
        torch.save(state_dict, path)

    def get_ica_modules(self):
        """
        Modules used for ICA scoring:
        - feature extractor (used by extract_feature)
        - CLCA
        - prediction head
        """
        return [self.backbone.extract_feature, self.clca, self.pred_head]
