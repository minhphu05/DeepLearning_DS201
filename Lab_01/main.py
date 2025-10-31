import numpy as np
import torch
from torch import nn

from mnist_dataset import collate_fn, MnistDataset
from torch.utils.data import DataLoader
from perceptron_1_layer import Perceptron_1_layer
from perceptron_3_layer import Perceptron_3_layer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


def train(dataloader: DataLoader) -> list:
    pass

def evaluate(model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
    model.eval()
    predictions = []
    trues = []
    
    with torch.no_grad():
        for item in dataloader:
            image: torch.Tensor = item["image"].to(device)
            label: torch.Tensor = item["label"].to(device)

            output: torch.Tensor = model(image)
            output = torch.argmax(output, dim=1)
            
            predictions.extend(output.cpu().tolist())
            trues.extend(label.cpu().tolist())
            
    return {
        "accuracy": accuracy_score(trues, predictions),
        "precision": precision_score(trues, predictions, average="weighted", zero_division=0),
        "all": recall_score(trues, predictions, average="weighted", zero_division=0),
        "f1": f1_score(trues, predictions, average="weighted", zero_division=0)
    }

def compute_score(labels: torch.Tensor, predictions: torch.Tensor) -> dict:
    pass

if __name__ == "__main__":
    train_dataset = MnistDataset(
        image_path="/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_01/data/raw_data/train-images.idx3-ubyte",
        label_path="/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_01/data/raw_data/train-labels.idx1-ubyte"
    )
    
    test_dataset = MnistDataset(
        image_path="/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_01/data/raw_data/t10k-images.idx3-ubyte",
        label_path="/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_01/data/raw_data/t10k-labels.idx1-ubyte"
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 1-layer MLP
    # model = Perceptron_1_layer(
    #     image_size=(28,28),
    #     num_labels=10
    # ).to(device)
    
    # 3-layer MLP
    model = Perceptron_3_layer(
        image_size=(28,28),
        num_classes=10
    ).to(device)
    
    loss_fn = nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=0.01
    )
    
    EPOCHS = 10
    best_score = 0
    best_score_name = "f1"
    for epoch in range (EPOCHS):
        print(f"Epoch {epoch}")
        losses = []
        model.train ()
        for item in train_dataloader:
            image = item["image"].to(device)
            label = item["label"].to(device)
            
            # forward pass
            output = model(image)
            loss = loss_fn(output, label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        print(f"Loss: {np.array(losses).mean()}")
        
        scores = evaluate(model, test_dataloader)
        for score_name, value in scores.items():
            print(f"{score_name}: {value:.4f}")
            
        # current_score = scores[best_score_name]
        # if current_score > best_score_name:
        #     best_score_name = current_score
        #     torch.save(
        #         model.parameters(),
        #         "./checkpoints/perceptron_1_layer/best_model.pth"
        #     )