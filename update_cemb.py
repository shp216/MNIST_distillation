import torch

def update_cemb(cemb_trainer, x,c,t,noise, feature_loss_weight, inversion_loss_weight, update_cemb_num, lr):
    T_cemb1=None
    T_cemb2=None
    for i in range(update_cemb_num):
        T_cemb1, T_cemb2, T_grad_cemb1, T_grad_cemb2 = cemb_trainer(x,c,t,noise, feature_loss_weight, inversion_loss_weight, T_cemb1, T_cemb2)
        with torch.no_grad():
            T_grad_cemb1 = T_grad_cemb1.detach()  # 그래프와의 연결을 끊음
            T_grad_cemb2 = T_grad_cemb2.detach()  # 그래프와의 연결을 끊음

            T_cemb1 -= lr * T_grad_cemb1
            T_cemb2 -= lr * T_grad_cemb2
    update_cemb1 = T_cemb1
    update_cemb2 = T_cemb2
    return update_cemb1, update_cemb2