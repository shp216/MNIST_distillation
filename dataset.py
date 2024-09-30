import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class using only cached data
class MNISTDataset_x0(Dataset):
    def __init__(self, cached_images, cached_labels):
        self.cached_images = cached_images
        self.cached_labels = cached_labels

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        img = self.cached_images[idx]
        label = self.cached_labels[idx]

        return img, label
    
class MNISTDataset(Dataset):
    def __init__(self, img_cache, t_cache, class_cache):
        self.img_cache = img_cache
        self.t_cache = t_cache
        self.class_cache = class_cache

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        img = self.img_cache[idx]
        t = self.t_cache[idx]
        label = self.class_cache[idx]

        return img, t, label, idx
    
    def update_data(self, indices, new_imgs):
        # 이 부분을 고쳐서 `indices`를 정수 배열 형태로 바꿔 인덱싱
        indices = indices.view(-1).long()  # ensure indices are a flat, long tensor
        
        device = self.img_cache.device
        indices = indices.to(device)
        new_imgs = new_imgs.to(device)
        
        # print('self.img_cache device:', self.img_cache.device)
        # print('indices device:', indices.device)
        # print('new_imgs device:', new_imgs.device)
        
        # 인덱스가 맞지 않는 경우 torch.index_select로 인덱스를 처리
        self.img_cache.index_copy_(0, indices, new_imgs)  # indices에 맞는 부분만 교체
        self.t_cache.index_copy_(0, indices, self.t_cache[indices] - 1)
        
        # t_cache 값이 0 미만인 인덱스 처리
        negative_indices = (self.t_cache[indices] < 0).nonzero(as_tuple=True)[0]
        
        # 실제 zero_indices를 전체 t_cache 기준으로 변환
        zero_indices = indices[negative_indices]
        num_zero_indices = zero_indices.size(0)

        if num_zero_indices > 0:
            # 0인 인덱스를 T-1로 초기화
            self.t_cache.index_fill_(0, zero_indices, 999)
            self.img_cache.index_copy_(0, zero_indices, torch.randn(
                num_zero_indices, 
                new_imgs.shape[1],  # channels
                new_imgs.shape[2],  # height
                new_imgs.shape[3],  # width
                device=device 
            ))