import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import logging
import os
import math
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score

from dataloader import ViOCD_Dataset, collate_fn
from vocab import Vocabulary
from module.classification import TransformerModel

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TRAIN_PATH = "/kaggle/input/viocd-dataset/train.json"
    DEV_PATH   = "/kaggle/input/viocd-dataset/dev.json"
    TEST_PATH  = "/kaggle/input/viocd-dataset/test.json"
    SAVE_PATH  = "/kaggle/working/transformer_base.pth"
    MIN_FREQ   = 1

    D_MODEL = 256       
    N_HEAD = 4         
    N_LAYERS = 2        
    D_FF = 1024         
    DROPOUT = 0.2      

    BATCH_SIZE = 64   
    EPOCHS = 30
    
    ADAM_BETAS = (0.9, 0.98) 
    ADAM_EPS = 1e-9          
    WARMUP_STEPS = 4000     

config = Config()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * \
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

# --- 3. TRAINING LOOP ---
def train(model, dataloader, optimizer_wrapper, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Training", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(config.DEVICE)
        label_ids = batch["label_ids"].to(config.DEVICE)

        optimizer_wrapper.zero_grad()
        
        _, loss = model(input_ids, label_ids)

        loss.backward()
        optimizer_wrapper.step()

        total_loss += loss.item()
        current_lr = optimizer_wrapper._rate
        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}", "lr": f"{current_lr:.6f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(config.DEVICE)
            label_ids = batch["label_ids"].to(config.DEVICE)
            
            logits, loss = model(input_ids, label_ids)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = label_ids.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    avg_loss = total_loss / len(dataloader)
    
    acc = accuracy_score(all_labels, all_preds)
    
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    return avg_loss, acc, f1, all_preds, all_labels

def main():
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Model Config: d_model={config.D_MODEL}, layers={config.N_LAYERS}, heads={config.N_HEAD}")
    
    if not os.path.exists(config.TRAIN_PATH):
        logger.error(f"Không tìm thấy file train: {config.TRAIN_PATH}")
        return

    logger.info("Building Vocabulary...")
    vocab = Vocabulary(config.TRAIN_PATH, min_freq=config.MIN_FREQ)
    logger.info(f"Vocab Size: {vocab.vocab_size} tokens")
    logger.info(f"Labels ({vocab.num_labels}): {vocab.label2idx}")

    train_dataset = ViOCD_Dataset(config.TRAIN_PATH, vocab)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    dev_loader = None
    if os.path.exists(config.DEV_PATH):
        dev_dataset = ViOCD_Dataset(config.DEV_PATH, vocab)
        dev_loader = DataLoader(
            dev_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=collate_fn
        )
    
    logger.info("Initializing Transformer Model (Base)...")
    model = TransformerModel(
        d_model=config.D_MODEL,
        head=config.N_HEAD,
        n_layers=config.N_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        vocab=vocab
    ).to(config.DEVICE)

    base_optimizer = optim.Adam(
        model.parameters(), 
        lr=0,
        betas=config.ADAM_BETAS, 
        eps=config.ADAM_EPS
    )
    
    optimizer = NoamOpt(config.D_MODEL, 2, config.WARMUP_STEPS, base_optimizer)

    logger.info("Start Training...")
    best_f1 = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, epoch)
        log_msg = f"Epoch {epoch}/{config.EPOCHS} | Train Loss: {train_loss:.4f}"
        
        if dev_loader:
            val_loss, val_acc, val_f1, _, _ = evaluate(model, dev_loader)
            log_msg += f" | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), config.SAVE_PATH)
                log_msg += " [BEST MODEL SAVED]"
        else:
            torch.save(model.state_dict(), config.SAVE_PATH)
            
        logger.info(log_msg)

    logger.info("Training Completed.")

    if os.path.exists(config.TEST_PATH):
        logger.info("Evaluating on Test Set...")
        
        if os.path.exists(config.SAVE_PATH):
            model.load_state_dict(torch.load(config.SAVE_PATH, map_location=config.DEVICE))
            logger.info(f"Loaded best model from {config.SAVE_PATH}")
        
        test_dataset = ViOCD_Dataset(config.TEST_PATH, vocab)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader)
        
        logger.info(f"TEST RESULTS | Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
        
        target_names = [vocab.idx2label[i] for i in range(len(vocab.label2idx))]
        report = classification_report(test_labels, test_preds, target_names=target_names)
        print("\nClassification Report:\n")
        print(report)

if __name__ == "__main__":
    main()