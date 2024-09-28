import torch
from tqdm import tqdm
from script import ContextUnet, DDPM
import argparse, sys, os

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pre_caching", action='store_true', help='only precaching')
    
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')
    
    
    
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
    
    
  
def distillation(args):
    device = torch.device('cuda:0')
    model_path = "./data/diffusion_outputs10/model_39.pth"  # Replace with your actual model path
    T_model = load_teacher_model(model_path)
    
    

def main(argv):
  
    parser = get_parser()
    args = parser.parse_args(argv[1:])
    
    if args.precaching:
        precaching(args)
        
    else:
        distillation(args)
  
if __name__ == "__main__":
    main(sys.argv)
    
