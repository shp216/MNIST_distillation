import torch
import torch.nn as nn
import torch.nn.functional as F
from gpu_log import GPUMonitor


class distillation_DDPM_trainer(nn.Module):
    
    def __init__(self, T_model, S_model, distill_features = False, inversion_loss=False):

        super().__init__()

        self.T_model = T_model
        self.S_model = S_model
        self.distill_features = distill_features
        self.inversion_loss = inversion_loss
        self.training_loss = nn.MSELoss()
    
        
    def forward(self, x0, c, t, noise, feature_loss_weight = 0.1, inversion_loss_weight=0.1):
        """
        Perform the forward pass for knowledge distillation.
        """
        ############################### TODO ###########################
        
        
        context_mask = torch.bernoulli(torch.zeros_like(c)+0.1).to(x0.device)
        
        if self.distill_features:
            # Teacher model forward pass (in evaluation mode)
            self.T_model.eval()
            if self.inversion_loss:
                T_output, T_features, T_cemb1, T_cemb2 = self.T_model(x0,c,t,noise,context_mask)
                
            else:
                with torch.no_grad():
                    #teacher_output, teacher_features = self.T_model.forward_features(x_t, t)
                    T_output, T_features, T_cemb1, T_cemb2 = self.T_model(x0,c,t,noise,context_mask)
                

            #student_output, student_features = self.S_model.forward_features(x_t, t)
            self.S_model.train()
            S_output, S_features, S_cemb1, S_cemb2 = self.S_model(x0,c,t,noise,context_mask)
                            
            output_loss = self.training_loss(S_output, T_output)
            
            feature_loss = 0
            for student_feature, teacher_feature in zip(S_features, T_features):
                feature_loss += self.training_loss(student_feature, teacher_feature)
                
            total_loss = output_loss + feature_loss_weight * feature_loss / len(S_features)
            
        else:
            self.T_model.eval()
            if self.inversion_loss:
                T_output, T_features, T_cemb1, T_cemb2 = self.T_model(x0,c,t,noise,context_mask)
                
            else:
                with torch.no_grad():
                    #teacher_output, teacher_features = self.T_model.forward_features(x_t, t)
                    T_output, T_features, T_cemb1, T_cemb2 = self.T_model(x0,c,t,noise,context_mask)
                
            
            self.S_model.train()
            S_output, S_features, S_cemb1, S_cemb2 = self.S_model(x0,c,t,noise,context_mask)
                
            output_loss = self.training_loss(S_output, T_output)
            total_loss = output_loss
        
        # T_cemb1.requires_grad_(True)
        # T_cemb2.requires_grad_(True)
        # S_cemb1.requires_grad_(True)
        # S_cemb2.requires_grad_(True)
        if self.inversion_loss:
            T_optimize_loss = self.training_loss(T_output,noise)
            S_optimize_loss = self.training_loss(S_output,noise)
            
            T_grad_cemb1 = torch.autograd.grad(T_optimize_loss, T_cemb1, create_graph=True)[0].detach()
            T_grad_cemb2 = torch.autograd.grad(T_optimize_loss, T_cemb2, create_graph=True)[0].detach()
            S_grad_cemb1 = torch.autograd.grad(S_optimize_loss, S_cemb1, create_graph=True)[0]
            S_grad_cemb2 = torch.autograd.grad(S_optimize_loss, S_cemb2, create_graph=True)[0]
            
            grad_loss1 = self.training_loss(S_grad_cemb1, T_grad_cemb1)
            grad_loss2 = self.training_loss(S_grad_cemb2, T_grad_cemb2)
            
            total_loss += inversion_loss_weight * (grad_loss1 + grad_loss2)
        
        return output_loss, total_loss
            
        