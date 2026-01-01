import json
import re
import os
import torch
import numpy as np
from pyvi import ViTokenizer
from typing import List, Dict, Tuple

class Vocabulary:
    def __init__(self, data_path: str, min_freq: int = 1):
        self.data_path = data_path
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        data_iter = raw if isinstance(raw, list) else raw.values()

        word_freq: Dict[str, int] = {}
        
        self.label_freq: Dict[str, int] = {} 
        
        for item in data_iter:
            if "review" not in item or "domain" not in item:
                continue
                
            sent = self._preprocess_sentence(item["review"])
            for w in sent.split():
                word_freq[w] = word_freq.get(w, 0) + 1
            
            label = str(item["domain"])
            self.label_freq[label] = self.label_freq.get(label, 0) + 1
        
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        sorted_words = sorted(
            [item for item in word_freq.items() if item[1] >= min_freq], 
            key=lambda x: (-x[1], x[0])
        )
        
        base_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        vocab_words = base_tokens + [w for w, _ in sorted_words]

        self.word2idx: Dict[str, int] = {w: i for i, w in enumerate(vocab_words)}
        self.idx2word: Dict[int, str] = {i: w for w, i in self.word2idx.items()}

        # Sắp xếp label để đảm bảo thứ tự index cố định
        sorted_labels = sorted(self.label_freq.keys())
        self.label2idx: Dict[str, int] = {l: i for i, l in enumerate(sorted_labels)}
        self.idx2label: Dict[int, str] = {i: l for l, i in self.label2idx.items()}

        self.pad_idx = self.word2idx[self.pad_token]
        self.bos_idx = self.word2idx[self.bos_token]
        self.eos_idx = self.word2idx[self.eos_token]
        self.unk_idx = self.word2idx[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def num_labels(self) -> int:
        return len(self.label2idx)

    def get_class_weights(self, device=None) -> torch.Tensor:
        """
        Tính trọng số cho từng class dựa trên tần suất xuất hiện.
        Class ít xuất hiện sẽ có trọng số cao hơn.
        Công thức: Weight = Total_Samples / (Num_Classes * Class_Count)
        """
        counts = [self.label_freq[self.idx2label[i]] for i in range(self.num_labels)]
        counts = np.array(counts)
        
        total_samples = sum(counts)
        n_classes = len(counts)
        
        weights = total_samples / (n_classes * counts)
        
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        
        if device:
            weights_tensor = weights_tensor.to(device)
            
        return weights_tensor

    def _preprocess_sentence(self, sentence: str) -> str:
        if not isinstance(sentence, str): return ""
        s = sentence.lower()
        s = re.sub(r"https?://\S+|www\.\S+", " ", s)
        s = re.sub(r"[^0-9a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = ViTokenizer.tokenize(s)
        return s

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        unk = self.word2idx[self.unk_token]
        return [self.word2idx.get(t, unk) for t in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.idx2word.get(i, self.unk_token) for i in ids]

    def encode_sentence(self, sentence: str, add_bos_eos: bool = True) -> List[int]:
        sent = self._preprocess_sentence(sentence)
        tokens = sent.split()
        ids = self.tokens_to_ids(tokens)
        if add_bos_eos:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def decode_ids(self, ids: List[int], remove_special: bool = True) -> str:
        tokens = self.ids_to_tokens(ids)
        if remove_special:
            specials = {self.pad_token, self.bos_token, self.eos_token}
            tokens = [t for t in tokens if t not in specials]
        return " ".join(tokens)

    def encode_batch(self, sentences: List[str], add_bos_eos: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.encode_sentence(s, add_bos_eos) for s in sentences]
        lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
        
        if len(lengths) == 0: return torch.empty(0), torch.empty(0)

        max_len = lengths.max().item()
        
        padded = torch.full((len(seqs), max_len), self.pad_idx, dtype=torch.long)
        for i, seq in enumerate(seqs):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded, lengths

    def encode_labels(self, labels: List[str]) -> torch.Tensor:
        return torch.tensor([self.label2idx[str(l)] for l in labels], dtype=torch.long)

    def decode_labels(self, label_ids: List[int]) -> List[str]:
        return [self.idx2label[i] for i in label_ids]