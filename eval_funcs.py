import os
import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from script import DDPM, ContextUnet
from classifier.model import SOPCNN
import gdown


    
### config ###
class Config:
    def __init__(self):
        self.batch_size = 1024
        self.learning_rate = 0.001
        self.epochs = 2000
        self.optimizer = 'adam'
        self.model_name = 'SOPCNN'
        self.num_classes = 10
        self.model_save_path = os.path.join('./classifier/MNIST_SOPCNN.pth')
        self.log_interval = 10
        self.patience = 5
        self.min_delta = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

def download_model_from_google_drive(drive_url, output_path):
    """
    Google Drive에서 모델 파일을 다운로드하는 함수.
    
    Args:
    - drive_url: Google Drive의 공유 링크 또는 파일 ID
    - output_path: 다운로드된 파일을 저장할 경로
    
    Returns:
    - output_path: 다운로드된 파일의 경로
    """
    gdown.download(drive_url, output_path, quiet=False, fuzzy=True)
    return output_path


### eval ###
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 폴더 구조에서 이미지 경로와 라벨 추출
        for class_label in range(10):  # 클래스가 10개라고 가정 (MNIST 기반)
            class_folder = os.path.join(root_dir, f'class_{class_label}')
            if os.path.exists(class_folder):
                for img_file in os.listdir(class_folder):
                    # .ipynb_checkpoints 같은 폴더나 잘못된 파일을 무시
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일만 포함
                        self.image_paths.append(os.path.join(class_folder, img_file))
                        self.labels.append(class_label)
        
        # 데이터 확인을 위한 디버깅 로그 추가
        # print(f"총 이미지 수: {len(self.image_paths)}, \n라벨 분포: \n{pd.Series(self.labels).value_counts()}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(image_path).float() / 255.0  # 이미지를 읽고 정규화
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def accuracy_per_class(preds, labels, num_classes):
    class_correct = [0. for _ in range(num_classes)]
    class_total = [0. for _ in range(num_classes)]
    
    for pred, label in zip(preds, labels):
        correct = pred == label
        class_correct[label] += correct
        class_total[label] += 1
    
    # 클래스 번호를 키로 하고 정확도를 값으로 가지는 딕셔너리 생성
    class_accuracies = {i: round(100 * class_correct[i] / class_total[i], 2) if class_total[i] > 0 else 0 
                    for i in range(num_classes)}
    
    return class_accuracies

def accuracy_seen_unseen(preds, labels, unseen_class_index, num_classes):
    """
    특정 클래스의 accuracy를 unseen으로 하고, 나머지 클래스들의 accuracy를 seen으로 계산.
    
    Args:
    - preds: 모델의 예측 값 리스트
    - labels: 실제 라벨 리스트
    - unseen_class_index: unseen으로 간주할 클래스 인덱스
    - num_classes: 전체 클래스 수
    
    Returns:
    - seen_accuracy: seen 클래스들의 평균 정확도
    - unseen_accuracy: unseen 클래스의 정확도
    """
    class_correct = [0. for _ in range(num_classes)]
    class_total = [0. for _ in range(num_classes)]
    
    for pred, label in zip(preds, labels):
        correct = pred == label
        class_correct[label] += correct
        class_total[label] += 1
    
    # Unseen 클래스의 정확도 계산
    unseen_correct = class_correct[unseen_class_index]
    unseen_total = class_total[unseen_class_index]
    unseen_accuracy = round(100 * unseen_correct / unseen_total, 2) if unseen_total > 0 else 0
    
    # Seen 클래스들의 정확도 계산 (Unseen 클래스를 제외한 나머지 클래스들)
    seen_correct = sum([class_correct[i] for i in range(num_classes) if i != unseen_class_index])
    seen_total = sum([class_total[i] for i in range(num_classes) if i != unseen_class_index])
    seen_accuracy = round(100 * seen_correct / seen_total, 2) if seen_total > 0 else 0
    
    return seen_accuracy, unseen_accuracy


# Test function to evaluate the model and get accuracy per class as a dictionary
def test(model, test_loader, config, unseen_class_index):
    model.eval()
    correct = 0
    device = config.device
    num_classes = 10
    preds = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            preds.extend(pred.cpu().numpy())
            labels.extend(target.cpu().numpy())

    test_accuracy = 100. * correct / len(test_loader.dataset)
    seen_accuracy, unseen_accuracy = accuracy_seen_unseen(preds, labels, unseen_class_index, num_classes)
    
    print(f'Total Accuracy: {test_accuracy:.2f}%')
    print(f'Seen Accuracy: {seen_accuracy:.2f}%')
    print(f'Unseen Accuracy: {unseen_accuracy:.2f}%')

    return test_accuracy, seen_accuracy, unseen_accuracy


### sampling per class ###

    
def sample_and_save_images(ddpm, w, n_classes, n_sample_per_class, save_dir, device):
    """
    각 클래스별로 지정된 수의 이미지를 샘플링하고 해당 클래스 폴더에 저장하는 함수
    Args:
    - ddpm: Diffusion 모델
    - w: guidance weight 값
    - n_classes: 클래스 수 (MNIST의 경우 10)
    - n_sample_per_class: 각 클래스당 샘플링할 이미지 수
    - save_dir: 이미지를 저장할 디렉토리 경로
    - device: 모델과 데이터를 처리할 디바이스 (예: 'cuda:0')
    """
    ddpm.eval()
    with torch.no_grad():
        n_sample_per_class = n_sample_per_class  # 클래스당 샘플링할 이미지 수
        for class_label in range(n_classes):
            x_gen, _ = ddpm._sample(n_sample_per_class, (1, 28, 28), device, guide_w=w, class_label=class_label)
    
            # 샘플링한 이미지를 저장할 디렉토리 생성
            save_dir_class = f"{save_dir}/class_{class_label}/"
            os.makedirs(save_dir_class, exist_ok=True)
            
            # 이미지 저장
            for i in range(n_sample_per_class):
                save_image(x_gen[i], f"{save_dir_class}/sample_{i}_w{w}.png")
    
            print(f"Class {class_label} - saved {n_sample_per_class} samples with w={w}")
    ddpm.train()


### sampling_eval

def sample_and_test_model(n_sample_per_class=100, w=0.0, save_dir='./output_samples', model="./model_39.pth", unseen_class_index=3):
    """
    DDPM 모델을 사용하여 이미지를 샘플링하고, 해당 샘플 데이터를 SOPCNN 모델로 테스트하는 함수.
    
    Args:
    - n_sample_per_class: 클래스당 샘플링할 이미지 수 (기본값: 100)
    - w: generative guidance 값 (기본값: 0.0)
    - save_dir: 샘플링한 이미지를 저장할 디렉토리 경로 (기본값: './output_samples')
    - model: DDPM 모델 가중치 경로 (기본값: '../model_39.pth')
    
    Returns:
    - test_accuracy: 전체 정확도
    - class_accuracies: 클래스별 정확도 리스트
    """
    
    # 설정값들
    device = config.device
    n_classes = 10
    
    # DDPM 모델 불러오기
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=128, n_classes=n_classes), betas=(1e-4, 0.02), n_T=400, device=device, drop_prob=0.1)
    ddpm.to(device)

    # model_path가 문자열(teacher)이면 해당 경로에서 모델 로드, 그렇지 않으면 이미 주어진 모델 사용
    if isinstance(model, str):
        print(f"모델을 경로 {model}에서 로드 중...")
        ddpm.load_state_dict(torch.load(model))
    else:
        print("이미 주어진 모델을 사용합니다.")
        ddpm = model

    
    # 샘플링 진행
    print("샘플링 중...")
    sample_and_save_images(ddpm, w, n_classes, n_sample_per_class, save_dir, device)

    # 데이터셋과 DataLoader 준비
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 1채널 (흑백)로 변환
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 정규화
    ])

    dataset = ImageFolderDataset(root_dir=save_dir, transform=transform)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)

    # SOPCNN 모델 불러오기
    print("classifier 모델 로딩 중...")
    model = SOPCNN(num_classes=config.num_classes).to(device)


    if not os.path.exists(config.model_save_path):
        drive_url = "https://drive.google.com/file/d/1C-7j17RwzdHMTqXrVRy185EHxlhbuMdZ/view?usp=drive_link"
        classifier_model_path = download_model_from_google_drive(drive_url, config.model_save_path)
        state_dict = torch.load(classifier_model_path)['model_state_dict']
    else:
        state_dict = torch.load(config.model_save_path)['model_state_dict']
    
    model.load_state_dict(state_dict)

    # 정확도 측정
    print("테스트 중...")
    test_accuracy, seen_accuracy, unseen_accuracy = test(model, test_loader, config, unseen_class_index)

    return test_accuracy, seen_accuracy, unseen_accuracy



if __name__ == "__main__":
    # test
    
    n_sample_per_class = 100  
    w = 0.0  
    save_dir = './output_samples'
    model_path = "./model_39.pth"
    unseen_class_index = 3  

    test_accuracy, seen_accuracy, unseen_accuracy = sample_and_test_model(
        n_sample_per_class, w, save_dir, model=model_path, unseen_class_index=unseen_class_index
    )

    print("###")
    print(test_accuracy)
    print("###")
    print(seen_accuracy)
    print("###")
    print(unseen_accuracy)
    