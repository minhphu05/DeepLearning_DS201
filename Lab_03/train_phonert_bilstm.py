import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from DataUtils.phoNERT import phoNERT, Vocab, collate_fn
from model.BiLSTM import BiLSTM
from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm import tqdm
import logging
import numpy as np
import os
from os import path

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("mps" if torch.mps.is_available() else "cpu")

def train(model: nn.Module, 
          data: DataLoader, 
          epoch: int, 
          loss_fn: nn.Module, 
          optimizer: optim.Optimizer) -> float:

    model.train()
    running_loss = []
    pbar = tqdm(data, desc=f"Epoch {epoch} - Training")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        tags_ids = batch["tags_ids"].to(device) 
        lengths = batch["lengths"] 

        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, lengths) 

        # Flatten output và labels để tính Loss
        # Logits: [Batch * Seq, Num_Tags]
        # Labels: [Batch * Seq]
        loss = loss_fn(logits.view(-1, logits.shape[-1]), tags_ids.view(-1))

        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        pbar.set_postfix({"loss": sum(running_loss)/len(running_loss)})
    
    return sum(running_loss)/len(running_loss)

def evaluate(model: nn.Module, data: DataLoader, epoch: int) -> float:
    model.eval()
    true_labels = []
    predictions = []

    pbar = tqdm(data, desc=f"Epoch {epoch} - Evaluation")
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            tags_ids = batch["tags_ids"].to(device)
            lengths = batch["lengths"]

            logits = model(input_ids, lengths)
            predicted_tags = torch.argmax(logits, dim=-1)

            # Lọc bỏ padding (-100) để tính điểm chính xác
            mask = tags_ids != -100
            
            valid_tags = tags_ids[mask].cpu().numpy()
            valid_preds = predicted_tags[mask].cpu().numpy()

            true_labels.extend(valid_tags)
            predictions.extend(valid_preds)
            
    # Tính toán Metrics
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)

    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info("----------------------------------")

    return f1

if __name__ == "__main__":
    data_dir = "/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_03/data/PhoNERT/syllable"
    output_dir = "/Users/kittnguyen/Downloads/Learning_Documents/Deep_Learning/Lab_03/model_result" 
    
    train_path = path.join(data_dir, "train_syllable.json")
    dev_path = path.join(data_dir, "dev_syllable.json")
    test_path = path.join(data_dir, "test_syllable.json")
    
    best_model_path = path.join(output_dir, "lstm_best_model.pt")

    logging.info(f"Device being used: {device}")

    logging.info("Loading vocab ... ")
    vocab = Vocab(filepath=train_path)

    logging.info("Loading dataset ... ")
    train_dataset = phoNERT(train_path, vocab=vocab)
    dev_dataset = phoNERT(dev_path, vocab=vocab)
    test_dataset = phoNERT(test_path, vocab=vocab)

    logging.info("Creating dataloader ... ")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    logging.info("Building Bi-LSTM NER model ... ")
    model = BiLSTM(
        vocab_size=vocab.vocab_size,
        embedding_dim=300, 
        hidden_size=256,
        num_tags=vocab.num_tags,
        num_layers=5,
        dropout=0.5,
        padding_idx=0
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 

    epoch = 0
    best_f1 = 0
    patience = 0
    patience_limit = 10
    
    logging.info("Starting training ...")
    
    while True:
        epoch += 1
        train_loss = train(model, train_dataloader, epoch, loss_fn, optimizer)
        f1 = evaluate(model, dev_dataloader, epoch)
        
        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best F1: {best_f1:.4f}. Saved model.")
        else:
            patience += 1
            logging.info(f"No improvement. Patience: {patience}/{patience_limit}")
        
        if ((patience == patience_limit) or (epoch == 100)): 
            logging.info("Stopping training.")
            break
                  
    logging.info("Loading best model for final test ...")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
          
    test_f1 = evaluate(model, test_dataloader, epoch)
    logging.info(f"Final F1 score on TEST set: {test_f1}")
