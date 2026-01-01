import torch
from torch.utils.data import DataLoader
from torch import optim
import logging
import os
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score

from dataset import PhoNER_COVID19, Vocab, collate_fn
from module.sequential_labeling import TransformerModel

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TRAIN_PATH = "/kaggle/input/phoner-covid19/train_word.json" 
    DEV_PATH   = "/kaggle/input/phoner-covid19/dev_word.json"     
    TEST_PATH  = "/kaggle/input/phoner-covid19/test_word.json"
    SAVE_PATH  = "ner_transformer.pth"
    MIN_FREQ   = 1

    D_MODEL = 256       
    N_HEAD = 4
    N_LAYERS = 3        
    D_FF = 1024
    DROPOUT = 0.1

    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4

config = Config()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def train(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Training", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(config.DEVICE)
        tags_ids = batch["tags_ids"].to(config.DEVICE) 

        optimizer.zero_grad()
        
        _, loss = model(input_ids, tags_ids)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, vocab):
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(config.DEVICE)
            tags_ids = batch["tags_ids"].to(config.DEVICE)
            
            logits, loss = model(input_ids, tags_ids)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            
            preds_np = preds.cpu().numpy()
            labels_np = tags_ids.cpu().numpy()

            mask = labels_np != -100
            
            all_preds.extend(preds_np[mask])
            all_labels.extend(labels_np[mask])
            
    avg_loss = total_loss / len(dataloader)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    return avg_loss, acc, f1, all_preds, all_labels

def main():
    logger.info(f"Using device: {config.DEVICE}")
    
    if not os.path.exists(config.TRAIN_PATH):
        logger.error(f"File not found: {config.TRAIN_PATH}")
        return

    logger.info("Building Vocabulary...")
    vocab = Vocab(config.TRAIN_PATH, min_freq=config.MIN_FREQ)
    logger.info(f"Vocab Size: {vocab.vocab_size}")
    logger.info(f"Tags ({vocab.n_tags}): {vocab.tag2idx}")

    train_dataset = PhoNER_COVID19(config.TRAIN_PATH, vocab)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    dev_dataset = PhoNER_COVID19(config.DEV_PATH, vocab)
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    logger.info("Initializing Transformer Model for NER...")
    model = TransformerModel(
        d_model=config.D_MODEL,
        head=config.N_HEAD,
        n_layers=config.N_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        vocab=vocab
    ).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    logger.info("Start Training...")
    best_f1 = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        # TRAIN
        train_loss = train(model, train_loader, optimizer, epoch)
        log_msg = f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {train_loss:.4f}"
        

        val_loss, val_acc, val_f1, _, _ = evaluate(model, dev_loader, vocab)
        log_msg += f" | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), config.SAVE_PATH)
            log_msg += " [BEST SAVED]"
        torch.save(model.state_dict(), config.SAVE_PATH)
            
        logger.info(log_msg)

    if config.TEST_PATH and os.path.exists(config.TEST_PATH):
        logger.info("Evaluating on Test Set...")
        
        if os.path.exists(config.SAVE_PATH):
            model.load_state_dict(torch.load(config.SAVE_PATH, map_location=config.DEVICE))
            logger.info("Loaded best model.")
        
        test_dataset = PhoNER_COVID19(config.TEST_PATH, vocab)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, vocab)
        logger.info(f"TEST | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

        unique_labels = sorted(list(set(test_labels)))
        target_names = [vocab.idx2tag.get(i, "UNK") for i in unique_labels]
        
        print("\nClassification Report:\n")
        print(classification_report(test_labels, test_preds, labels=unique_labels, target_names=target_names))

if __name__ == "__main__":

    main()