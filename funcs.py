import os
import random
import torch
import torch.nn.init as init
from torchvision.utils import save_image, make_grid
import numpy as np
from einops import rearrange
from PIL import Image
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import wandb
import torch.distributed as dist
import shutil
import time
from script import ContextUnet, DDPM



from ldm.util import instantiate_from_config



def load_teacher_model(model_path, device="cuda:0"):
    # Define the model architecture that matches the saved model
    n_classes = 10
    n_feat = 128  # or whatever value you used during training
    n_T = 400     # or whatever value you used during training

    # Initialize the same model structure
    teacher_model = DDPM(
        nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), 
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        drop_prob=0.1
    ).to(device)

    # Load the saved state dictionary into the model
    teacher_model.load_state_dict(torch.load(model_path, map_location=device))
    teacher_model.eval()  # Set the model to evaluation mode
    print(f"Teacher Model loaded:  loaded from {model_path}")
    
    return teacher_model

def load_student_model(device="cuda:0"):
    # Define the model architecture that matches the saved model
    n_classes = 10
    n_feat = 128  # or whatever value you used during training
    n_T = 400     # or whatever value you used during training
    student_model = DDPM(
        nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), 
        betas=(1e-4, 0.02), 
        n_T=n_T, 
        device=device, 
        drop_prob=0.1
    ).to(device)
    
    student_model.train()
    print(f"Student Model loaded:  without ckpt")
    
    return student_model

def load_pretrained_weights(S_model, checkpoint_path):
    # 전체 모델의 state_dict를 불러옵니다.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # student 모델의 nn_model의 현재 state_dict를 가져옵니다.
    model_state_dict = S_model.nn_model.state_dict()
        
    # 필요한 레이어만 업데이트합니다.
    layers_to_load = ['timeembed1', 'timeembed2', 'contextembed1', 'contextembed2']
    
    for layer_name in layers_to_load:
        # 해당 레이어의 가중치 키들을 가져옵니다.
        pretrained_layer_state = {k: v for k, v in checkpoint.items() if layer_name in k}
        
        # 해당 레이어의 가중치만 업데이트합니다.
        model_state_dict.update(pretrained_layer_state)
        
        # nn_model에서 해당 레이어 가져와서 requires_grad를 False로 설정
        layer = getattr(S_model.nn_model, layer_name)
        for param in layer.parameters():
            param.requires_grad = False
    
    # 업데이트된 state_dict로 모델에 로드합니다.
    S_model.nn_model.load_state_dict(model_state_dict, strict=False)  # strict=False로 해서 일부만 로드될 수 있도록 설정
    print(f"Loaded weights and frozen for layers: {', '.join(layers_to_load)}")



def save_checkpoint(S_model, optimizer, step, logdir):
    ckpt = {
        'student_model': S_model.state_dict(),  # S_model의 상태 저장
        #'scheduler': lr_scheduler.state_dict(),  # 학습 스케줄러 상태 저장
        'optimizer': optimizer.state_dict(),  # 옵티마이저 상태 저장
        'step': step,  # 현재 스텝 저장
    }
    
    # 저장 경로 설정
    save_path = os.path.join(logdir, f'student_ckpt_step_{step}.pt')
    
    # 디렉터리 존재 여부 확인 후 생성
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    
    # 체크포인트 저장
    torch.save(ckpt, save_path)
    print(f"Checkpoint saved at step {step} to {save_path}")



def sample_images(S_model, num_save_image, save_dir, step, device):
    x_gen, _ = S_model.sample(num_save_image, (1, 28, 28), device, guide_w=2.0)
    x_gen = (x_gen * -1 + 1)  # 이미지 범위를 [0, 1]로 변환
    x_gen_tensor = torch.tensor(x_gen) if not isinstance(x_gen, torch.Tensor) else x_gen
    grid_T = make_grid(x_gen_tensor, nrow=10) 
    save_image(grid_T, os.path.join(save_dir, f"sample_image_step_{step}.png"))
    print(f"save sample_image_step_{step}.png in {save_dir}")
