import torch
import torch.nn as nn
import math
from .transformer import TransformerEncoder, PositionnalEncoding, generate_padding_mask

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float, vocab):
        super(TransformerModel, self).__init__()

        self.vocab = vocab
        self.d_model = d_model 
        self.pad_idx = vocab.pad_idx

        self.embedding = nn.Embedding(vocab.vocab_size, d_model, padding_idx=self.pad_idx)
        self.PE = PositionnalEncoding(d_model, dropout)
        
        self.encoder = TransformerEncoder(d_model, head, n_layers, d_ff, dropout)

        self.ln_head = nn.Linear(d_model, vocab.num_labels)
        self.dropout = nn.Dropout(dropout)

        class_weights = vocab.get_class_weights() 
        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights, 
            ignore_index=self.pad_idx,
            label_smoothing=0.1
        )

        self._init_weights()

    def _init_weights(self):
        """Khởi tạo Xavier Uniform giúp hội tụ nhanh hơn"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        attention_mask = generate_padding_mask(input_ids, self.pad_idx).to(input_ids.device)

        input_embs = self.embedding(input_ids) * math.sqrt(self.d_model)
        features = self.PE(input_embs)
        
        features = self.encoder(features, attention_mask)

        mask_bool = (input_ids != self.pad_idx).unsqueeze(-1).float()
        
        masked_features = features * mask_bool
        
        sum_features = torch.sum(masked_features, dim=1) 

        count_tokens = torch.sum(mask_bool, dim=1).clamp(min=1e-9)
        
        pooled_features = sum_features / count_tokens

        logits = self.dropout(self.ln_head(pooled_features))
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return logits, loss