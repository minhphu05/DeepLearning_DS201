import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Việc sử dụng DataLoader from scratch khiến cho việc training bị tràn RAM nên sử dụng class có sẵn

def get_data_loaders(
    batch_size: int = 32,
    train_dir: str = "/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_02/Data/VinaFood21/train",
    test_dir: str = "/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_02/Data/VinaFood21/test"
):
    """
    Trả về train_loader, test_loader và số lớp (num_classes)
    """

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # chuyển ảnh có alpha channel về RGB
        transforms.Resize((224, 224)),                      # kích thước chuẩn
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)
    print(f"Loaded dataset: {num_classes} classes, {len(train_dataset)} train images, {len(test_dataset)} test images")

    return train_loader, test_loader, num_classes
