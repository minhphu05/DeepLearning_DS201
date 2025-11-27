import torch
from torch import nn

class GRU(nn.Module):
    def __init__(
        self,
        num_labels: int,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 5,
        padding_idx: int = 0,
        **kwargs: any
    ) -> None:
        
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            # dropout=0.3
        )
        
        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=num_labels
        )
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedding_features = self.embedding(inputs)
        features, _ = self.gru(embedding_features)
        feature = features[:,-1]
        outputs = self.classifier(feature)
        return outputs
        