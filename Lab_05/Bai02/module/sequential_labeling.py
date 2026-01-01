import torch
import torch.nn as nn
import math

from .transformer import TransformerEncoder, PositionnalEncoding, generate_padding_mask

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float, vocab):
        super(TransformerModel, self).__init__()

        self.vocab = vocab
        self.d_model = d_model 

        self.pad_token_idx = vocab.word2idx.get(vocab.pad, 0)

        self.embedding = nn.Embedding(vocab.vocab_size, d_model, padding_idx=self.pad_token_idx)
        self.PE = PositionnalEncoding(d_model, dropout)
        
        self.encoder = TransformerEncoder(d_model, head, n_layers, d_ff, dropout)

        self.ln_head = nn.Linear(d_model, vocab.n_tags)
        self.dropout = nn.Dropout(dropout)      
       
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        attention_mask = generate_padding_mask(input_ids, self.pad_token_idx).to(input_ids.device)
        
        input_embs = self.embedding(input_ids) * math.sqrt(self.d_model)
        features = self.PE(input_embs)
        
        features = self.encoder(features, attention_mask)
        
        logits = self.dropout(self.ln_head(features)) # [Batch, Seq_Len, Num_Tags]

        loss = None

        loss = self.loss_fn(logits.reshape(-1, self.vocab.n_tags), labels.reshape(-1))

        return logits, loss