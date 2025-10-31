from mnist_dataset import MnistDataset, collate_fn
from torch.utils.data import DataLoader
from torch import nn 
import torch
import numpy as np 
from LeNet import LeNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
# train_dataset = MnistDataset(
#     image_path="train-images.idx3-ubyte",
#     label_path="train-labels.idx1-ubyte"
# )

print("Hay nhap loai mo hinh (1 hoac 3): ")
model_type = input().strip()
if model_type == "1":
    model = LeNet(
        n_classes=10
    ).to(device)
elif model_type == "3":
    pass

loss_fn= nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle = True,
    collate_fn = collate_fn
)

def evaluate(model, dataloader):
    model.eval() 
    outputs = []
    trues = []
    for item in dataloader:
        image = item["image"].to(device) # (B, 28, 28)
        label = item["label"].to(device) # (B,)
        output = model(image)   # (B, 10) - raw logits
        predictions = torch.argmax(output, dim=-1)  # (B,) - predicted classes

        outputs.extend(predictions.tolist())
        trues.extend(label.tolist())
    return {
        "recall": recall_score(np.array(trues), np.array(outputs), average="macro"),
        "precision": precision_score(np.array(trues), np.array(outputs), average="macro"),
        "f1": f1_score(np.array(trues), np.array(outputs), average="macro"),
    }




EPOCHS = 10 
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch+1}")

    losses = []
    model.train() 
    for item in train_dataloader:
        image = item["image"].to(device) # (B, 28, 28)
        label = item["label"].to(device) # (B,)
        # Forward pass
        output = model(image)   # (B, 10)

        loss = loss_fn(output, label.long())
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {(np.array(losses).mean())}")
    metrics = evaluate(model, train_dataloader)
    for metric in metrics:
        print(f"{metric}: {metrics[metric]}")

