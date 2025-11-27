import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
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
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=0.5
        )
        
        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=num_labels
        )
    
    def forward(self, inputs: torch.Tensor):
        """
            inputs: torch.LongTensor(bs, len)
        """
        embedding_features = self.embedding(inputs)
        features, _ = self.lstm(embedding_features)
        
        # only get the last hidden size
        feature = features[:,-1]
        logits = self.classifier(feature)
        return logits
    
        