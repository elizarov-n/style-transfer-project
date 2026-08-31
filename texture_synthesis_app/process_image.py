import torch
from torchvision import transforms

mean_normalization = [0.485, 0.456, 0.406]
std_normalization = [0.229, 0.224, 0.225]

def normalization(image : torch.tensor):
    return transforms.Normalize(mean=mean_normalization, std=std_normalization)(image)

def denormalization(image: torch.tensor):
    device = image.device.type

    mean = torch.tensor(mean_normalization).view(3, 1, 1).to(device)
    std = torch.tensor(std_normalization).view(3, 1, 1).to(device)

    return (image * std + mean).clamp(0, 1)