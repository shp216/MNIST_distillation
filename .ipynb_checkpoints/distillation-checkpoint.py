import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid

from tqdm import tqdm, trange
from script import ContextUnet, DDPM
import argparse, sys, os
import time
from PIL import Image
import numpy as np
from trainer import distillation_DDPM_trainer
from dataset import MNISTDataset
from torch.utils.data import Dataset, DataLoader
from funcs import save_checkpoint, load_student_model, load_teacher_model, sample_images, load_pretrained_weights, show_images, visualize_t_cache_distribution
import warnings
from eval_funcs import sample_and_test_model
import logging

# 특정 경고 메시지를 무시
warnings.filterwarnings("ignore", category=FutureWarning)

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pre_caching", action='store_true', help='only precaching')
    parser.add_argument("--pre_caching_x0", action='store_true', help='only x0 precaching')
    
    parser.add_argument("--inversion_loss", action='store_true', help='using inversion_loss')
    parser.add_argument("--distill_features", action='store_true', help='perform knowledge distillation using intermediate features')


    
    parser.add_argument("--batch_size", type=int, default=256, help='batch size')
    parser.add_argument("--num_per_class", type=int, default=30, help='number of classes in MNIST')

    parser.add_argument("--n_T", type=int, default=400, help='number of timesteps in diffusion step')
    parser.add_argument("--n_classes", type=int, default=10, help='number of classes in MNIST')
    parser.add_argument("--n_sample", type=int, default=60000, help='number of total samples in MNIST')
    parser.add_argument("--cache_n", type=int, default=60000, help='number of cache data')
    parser.add_argument("--caching_batch_size", type=int, default=512, help='caching batch size')
    parser.add_argument("--num_save_image", type=int, default=50, help='number of total save images in save_step')


    parser.add_argument("--save_step", type=int, default=50, help='number of cache data')
    parser.add_argument("--sample_step", type=int, default=5, help='number of cache data')

    parser.add_argument("--logdir", type=str, default='./logs/', help='log directory')
    parser.add_argument("--save_dir", type=str, default='./images/', help='save directory')
    parser.add_argument("--eval_dir", type=str, default='./save_images/', help='eval image directory')
    parser.add_argument("--cache_dir", type=str, default='./cache/', help='cache directory')


    parser.add_argument("--ws_test", type=float, nargs='+', default=[0.0, 0.5, 2.0], help='List of values for ws_test')
    
    parser.add_argument("--n_epoch", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--feature_loss_weight", type=float, default=0.1, help="feature loss weighting")
    parser.add_argument("--inversion_loss_weight", type=float, default=0.1, help="inversion loss weighting")

    # eval
    parser.add_argument("--eval_step", type=int, default=50, help='eval step')
    parser.add_argument("--n_sample_per_class", type=int, default=100, help='number of samples per class')
    parser.add_argument("--w", type=float, default=0.0, help='generative guidance value')
    parser.add_argument("--save_samples_dir", type=str, default='./output_samples', help='directory to save output samples')
    parser.add_argument("--model_path", type=str, default="./model_39.pth", help='path to the model file')
    parser.add_argument("--unseen_class_index", type=int, default=3, help='index of the unseen class')


    return parser

def precaching(args):
    device = torch.device('cuda:0')
    model_path = "./model_39.pth"  # Replace with your actual model path
    T_model = load_teacher_model(model_path, args.n_T)
    T_model.to(device)
    
    n_T = args.n_T
    cache_size = args.cache_n
    cache_per_timestep = int(cache_size/n_T)
    
    img_cache = torch.zeros((cache_size, 1, 28, 28), dtype=torch.float32, device=device)  # MNIST 이미지 크기
    t_cache = torch.ones((cache_size,), dtype=torch.float32, device=device)*(n_T-1)
    
    selected_tensor = torch.tensor([0,1,2,4,5,6,7,8,9], device=device)
    class_cache = selected_tensor[torch.randint(0, len(selected_tensor), (cache_size,), device=device)]
    
    with torch.no_grad():
        indices = []
        
        for i in range(1,int(n_T/2)):
            # 0부터 i*n까지의 값
            indices.extend(range(i * cache_per_timestep))
            
            # (1000-i)*n부터 500*n까지의 값
            indices.extend(range((n_T - i) * cache_per_timestep-1, int(n_T/2) * cache_per_timestep-1, -1))
            
        for i in range(int(n_T/2)):
            indices.extend(range(int(n_T/2) * cache_per_timestep))          
                        
        # Batch size만큼의 인덱스를 뽑아오는 과정
        for batch_start in trange(0, len(indices), args.caching_batch_size, desc="Pre-caching"):
            batch_end = min(batch_start + args.caching_batch_size, len(indices))  # 인덱스 범위를 벗어나지 않도록 처리
            batch_indices = indices[batch_start:batch_end]  # Batch size만큼 인덱스 선택

            # 인덱스를 이용해 배치 선택
            img_batch = img_cache[batch_indices]
            t_batch = t_cache[batch_indices]
            class_batch = class_cache[batch_indices]

            x_prev, _ = T_model.cache_step(img_batch, class_batch, t_batch)

            # 결과를 저장
            img_cache[batch_indices] = x_prev
            t_cache[batch_indices] -= 1

            if batch_start % 1000 == 0:  # 예를 들어, 100 스텝마다 시각화
                visualize_t_cache_distribution(t_cache, cache_per_timestep)
                
        visualize_t_cache_distribution(t_cache, cache_per_timestep)
        
        save_dir = f"./{args.cache_dir}"  # 이미지와 레이블을 저장할 경로
        os.makedirs(save_dir, exist_ok=True)

        # img_cache와 class_cache를 .pt 파일로 저장
        torch.save(img_cache, os.path.join(save_dir, f"mnist_images.pt"))
        torch.save(t_cache, os.path.join(save_dir, f"mnist_t.pt"))
        torch.save(class_cache, os.path.join(save_dir, f"mnist_labels.pt"))

        print(f"Saved MNIST images, timestep and labels to {save_dir}")
        
def precaching_x0(args):
    device = torch.device('cuda:0')
    model_path = "./model_39.pth"  # Replace with your actual model path
    T_model = load_teacher_model(model_path, args.n_T)
    
    # S_model
    S_model = load_student_model(args.n_T)
    # optimizer
    
    if not os.path.exists(args.eval_dir):
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Created directory: {args.eval_dir}")
        
    with torch.no_grad():
        n_sample = args.n_sample  # 각 w마다 생성할 이미지 수 (예: 20000)
        total_samples = n_sample * len(args.ws_test)  # 전체 생성할 이미지 수 (예: 60000)
        
        save_dir = f"./{args.cache_dir}"  # 이미지와 레이블을 저장할 경로
        os.makedirs(save_dir, exist_ok=True)

        # 전체 이미지를 저장할 수 있도록 텐서 크기를 조정
        img_cache = torch.zeros((total_samples, 1, 28, 28), dtype=torch.float32, device=device)  # MNIST 이미지 크기
        class_cache = torch.zeros(total_samples, dtype=torch.long, device=device)  # 클래스 레이블

        # 전체 이미지를 생성하는 루프
        total_generated = 0
        for w_i, w in enumerate(args.ws_test):
            # exclude_sample에서 이미지와 해당 레이블을 반환받음
            x_gen, labels = T_model.exclude_sample(n_sample, (1, 28, 28), device, guide_w=w)
            
            #x_gen = (x_gen * -1 + 1)  # 이미지 범위를 [0, 1]로 변환
            x_gen_tensor = torch.tensor(x_gen) if not isinstance(x_gen, torch.Tensor) else x_gen

            # 생성된 이미지를 img_cache에 저장
            img_cache[total_generated:total_generated + x_gen_tensor.size(0)] = x_gen_tensor
            # 반환된 레이블을 class_cache에 저장
            class_cache[total_generated:total_generated + x_gen_tensor.size(0)] = labels
            
            total_generated += x_gen_tensor.size(0)

        # img_cache와 class_cache를 .pt 파일로 저장
        torch.save(img_cache, os.path.join(save_dir, f"mnist_images_x0.pt"))
        torch.save(class_cache, os.path.join(save_dir, f"mnist_labels_x0.pt"))

        print(f"Saved MNIST images and labels to {save_dir}")
        
        # 각 클래스별로 10개씩 이미지를 저장
        unseen_classes = [3]
        seen_classes = [cls for cls in range(args.n_classes) if cls not in unseen_classes]
        for cls in seen_classes:
            # 해당 클래스에 해당하는 이미지를 필터링
            class_indices = (class_cache == cls).nonzero(as_tuple=True)[0]
            
            # 해당 클래스에서 최대 10개까지 선택
            selected_indices = class_indices[:10]
            
            if len(selected_indices) == 0:
                print(f"No samples found for class {cls}. Skipping this class.")
                continue  # 이 클래스를 건너뜀
            
            if len(selected_indices) < 10:
                print(f"Class {cls} has fewer than 10 samples. Showing {len(selected_indices)} samples instead.")
            
            # 선택된 이미지를 추출
            selected_images = img_cache[selected_indices]

            # 이미지 그리드 생성 및 저장
            grid_T = make_grid(selected_images, nrow=5)  # 10개를 5개의 열로 배치
            save_image(grid_T, os.path.join(save_dir, f"class_{cls}_grid.png"))
        
def distillation_x0(args):
    device = torch.device('cuda:0')
    model_path = "./model_39.pth"  # Replace with your actual model path
    T_model = load_teacher_model(model_path, args.n_T)
    
    # S_model
    S_model = load_student_model(args.n_T)
    load_pretrained_weights(S_model, model_path)
    # optimizer
    optim = torch.optim.Adam(S_model.parameters(), lr=args.lr)
    # cache dataset
    
    # Dataset, DataLoader 설정
    img_cache_path = f"./{args.cache_dir}/mnist_images_x0.pt"
    label_cache_path = f"./{args.cache_dir}/mnist_labels_x0.pt"
    img_cache = torch.load(img_cache_path)
    class_cache = torch.load(label_cache_path)
    
    cache_dataset = MNISTDataset(img_cache, class_cache)
    dataloader = DataLoader(cache_dataset, batch_size=args.batch_size, shuffle=True)

    
    # trainer 설정
    trainer = distillation_DDPM_trainer(T_model, S_model, args.distill_features, args.inversion_loss)

    
    # training Loop
    for ep in range(args.n_epoch):
        print(f'epoch {ep}')
        S_model.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = args.lr*(1-ep/args.n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            t = torch.randint(1, args.n_T+1, (x.shape[0],)).to(device)  # t ~ Uniform(0, n_T)
            x = (x * -1 + 1)
            noise = torch.randn_like(x)
            output_loss, total_loss = trainer(x,c,t,noise, args.feature_loss_weight, args.inversion_loss_weight)
            total_loss.backward()
            pbar.set_description(f"loss: {total_loss:.4f}")
            optim.step()
    
        if ep % args.save_step == 0:
            save_checkpoint(S_model, optim, ep, args.logdir)      
            
        if ep % args.sample_step == 0:
            S_model.eval()
            sample_images(S_model, args.num_save_image, args.save_dir, ep, device)

        ## eval ##
        if ep % args.eval_step == 0:
            logging.basicConfig(filename='./logs/classifier_eval.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            S_model.eval()
        
            test_accuracy, seen_accuracy, unseen_accuracy = sample_and_test_model(args.n_sample_per_class, args.w, args.save_samples_dir, model=S_model, unseen_class_index=args.unseen_class_index)
            logging.info(f"Epoch {ep}: Total Accuracy: {test_accuracy}, Seen Accuracy: {seen_accuracy}, Unseen Accuracy: {unseen_accuracy}")

    # print("img_cache shape:", img_cache.shape)
    # print("class_cache shape:", class_cache.shape)
    
    # # 100개의 이미지를 가져옵니다
    # example_images = img_cache[:100]  # img_cache의 처음 100개 이미지를 선택

    # # 그리드 형태로 만들기 (10 x 10)
    # grid_image = make_grid(example_images, nrow=10)

    # # 이미지 저장 경로
    # save_dir = "./example_saved_images"
    # os.makedirs(save_dir, exist_ok=True)
    # save_image_path = os.path.join(save_dir, "example_100_images.png")

    # # 이미지 저장
    # save_image(grid_image, save_image_path)

    # print(f"Saved 100 images in grid form to {save_image_path}")
        

    

def main(argv):
  
    parser = get_parser()
    args = parser.parse_args(argv[1:])
    
    if args.pre_caching_x0:
        precaching_x0(args)

    if args.pre_caching:
        precaching(args)

    else:
        distillation_x0(args)
  
if __name__ == "__main__":
    main(sys.argv)
    
