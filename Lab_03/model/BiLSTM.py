import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int,
        num_layers: int = 5,
        hidden_size: int = 256,
        padding_idx: int = 0,
        dropout: float = 0.3,
        **kwargs: any 
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(
            in_features=hidden_size * 2,
            out_features=num_tags
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor
    ) -> None:
        emb = self.dropout(self.embedding(input_ids))
        
        packed_input = pack_padded_sequence(
            emb, 
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, _ = self.bilstm(packed_input)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True
        )
        
        logits = self.output_layer(self.dropout(output))
        return logits