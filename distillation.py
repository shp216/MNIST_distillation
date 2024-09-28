import torch
from torchvision.utils import save_image, make_grid

from tqdm import tqdm
from script import ContextUnet, DDPM
import argparse, sys, os

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pre_caching", action='store_true', help='only precaching')
    
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    
    
    parser.add_argument("--logdir", type=str, default='./logs', help='log directory')
    parser.add_argument("--savedir", type=str, default='./images', help='save directory')
    
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
    print(f"Model loaded from {model_path}")
    
    return teacher_model

def precaching(args):
    device = torch.device('cuda:0')
    model_path = "./data/diffusion_outputs10/model_39.pth"  # Replace with your actual model path
    T_model = load_teacher_model(model_path)
    T_model.to(device)
    
    ws_test = [0.0, 0.5, 2.0]
    n_classes = 10
    
    
    T_model.eval()
    with torch.no_grad():
        n_sample = 4*n_classes
        for w_i, w in enumerate(ws_test):
            x_gen, x_gen_store = T_model.sample(n_sample, (1, 28, 28), device, guide_w=w)
                        
            grid = make_grid(x_gen*-1 + 1, nrow=10)
            save_image(grid, args.save_dir + f"image_w{w}.png")
            print('saved image at ' + args.save_dir + f"image_w{w}.png")

    # add sampler (use DDPM class)
    
    # cache tensor
    
    # cache update
    
  
def distillation(args):
    device = torch.device('cuda:0')
    model_path = "./data/diffusion_outputs10/model_39.pth"  # Replace with your actual model path
    T_model = load_teacher_model(model_path)
    
    # S_model
    
    # optimizer
    
    # cache dataset
    
    # traning loop
    
    

def main(argv):
  
    parser = get_parser()
    args = parser.parse_args(argv[1:])
    
    if args.precaching:
        precaching(args)
        
    else:
        distillation(args)
  
if __name__ == "__main__":
    main(sys.argv)
    
