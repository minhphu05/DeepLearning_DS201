from os import path
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from DataUtils.uit_vsfc import UIT_VSFC, Vocab, collate_fn
from model.LSTM import LSTM
from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm import tqdm
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')

device = torch.device("mps" if torch.mps.is_available() else "cpu")

def train(model: nn.Module, 
          data: DataLoader, 
          epoch: int, 
          loss_fn: nn.Module, 
          optimizer: optim.Optimizer) -> None:

    model.train()
    running_loss = []

    pbar = tqdm(data, desc=f"Epoch {epoch} - Training")

    for batch in pbar:
        # batch là 1 dict (KHÔNG được for item in batch)
        input_ids = batch["input_ids"].to(device)
        labels = batch["label_ids"].to(device)

        logits = model(input_ids)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        pbar.set_postfix({"loss": sum(running_loss)/len(running_loss)})


def evaluate(model: nn.Module, data: DataLoader, epoch: int) -> float:
    model.eval()
    true_labels = []
    predictions = []

    pbar = tqdm(data, desc=f"Epoch {epoch} - Evaluation")
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label_ids"].to(device)

            logits = model(input_ids)
            
            predicted_labels = torch.argmax(logits, dim=-1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted_labels.cpu().numpy())
            
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)

    # Logging kết quả
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info("----------------------------------")

    # Trả về F1-score để sử dụng cho logic early stopping
    return f1

if __name__ == "__main__":
    logging.info("Loading vocab ... ")
    vocab = Vocab(root_dir="/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_03/data/UIT_VSFC")
    logging.info("Loading dataset ... ")

    train_dataset = UIT_VSFC(
        "/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_03/data/UIT_VSFC/UIT-VSFC-train.json",
        vocab=vocab
    )

    dev_dataset = UIT_VSFC(
        "/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_03/data/UIT_VSFC/UIT-VSFC-dev.json",
        vocab=vocab
    )

    test_dataset = UIT_VSFC(
        "/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_03/data/UIT_VSFC/UIT-VSFC-test.json",
        vocab=vocab
    )
    logging.info("Creating dataloader ... ")

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    logging.info("Building model ... ")

    model = LSTM(
        vocab_size=vocab.num_words,
        hidden_size=256,
        num_layers=5,
        num_labels=vocab.num_labels,
        padding_idx=0
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epoch = 0
    best_f1 = 0
    previous_f1 = 0
    patience = 0
    best_model_path = "/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_03/model_result/best_model.pt"

    while True:
        epoch += 1
        train(model, train_dataloader, epoch, loss_fn, optimizer)
        f1 = evaluate(model, dev_dataloader, epoch)
        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1
        
        current_f1 = f1
        if ((patience == 10) or (epoch == 200)):
            logging.info("Patience exceeded. Stopping training.")
            break
                  
    logging.info("Loading best model for final test ...")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
          
    test_f1 = evaluate(model, test_dataloader, epoch)
    logging.info(f"F1 score on test set: {test_f1}") 
