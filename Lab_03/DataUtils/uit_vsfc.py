import os
import json
import string
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

def collate_fn(items: dict) -> torch.Tensor:
    input_ids = [item["input_ids"] for item in items]
    max_len = max(len(ids) for ids in input_ids)
    input_ids = [
        F.pad(
            input,
            pad = (0, max(0, max_len - input.shape[0])),
            mode = "constant",
            value = 0
        ).unsqueeze(0) for input in input_ids]
    
    label_ids = [item["label_ids"].unsqueeze(0) for item in items]

    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "label_ids": torch.cat(label_ids, dim=0)
    }
    

class Vocab:
    def __init__(
        self, 
        root_dir: str
    ) -> None:
        all_words = set()
        labels = set()
        for filename in os.listdir(root_dir):
            data = json.load(open(os.path.join(root_dir, filename)))
            for item in data:
                sentence: str = item["sentence"]
                sentence = self._preprocess_sentence(sentence)
                all_words.update(sentence.split())
                labels.add(item["topic"])
        
        self.bos = "<s>"    # Begin of Sentence : Token đặc biệt đánh dấu bắt đầu câu
        self.pad = "<p>"    # Padding Token : Dùng để đệm cho các câu ngắn hơn để tất cả các câu trong batch đều có cùng độ dài
        # Ví dụ: 
            # Tôi ăn cơm <p>
            # Tôi ngủ <p> <p>
        
        """
        Ví dụ:
            self.word2idx = {
                "<p>": 0,
                "<s>": 1,
                "tôi": 2,
                "ăn": 3,
                "cơm": 4,
                "ngủ": 5
            }
        """
        self.word2idx = {
            word: idx for idx, word in enumerate(all_words, start=2)
        }                       
        self.word2idx[self.bos] = 1
        self.word2idx[self.pad] = 0
        
        """
            self.idx2word = {
                0: "<p>",
                1: "<s>",
                2: "tôi",
                3: "ăn",
                4: "cơm",
                5: "ngủ"
            }
        """
        self.idx2word = {
            idx: word for word, idx in self.word2idx.items()
        }
        
        self.label2idx = {
            label: idx for idx, label in enumerate(labels)
        }
        self.idx2label = {
            idx: label for label, idx in self.label2idx.items()
        }
    
    @property
    def num_labels(self) -> int:
        return len(self.label2idx)
    
    @property
    def num_words(self) -> int:
        return len(self.word2idx)
    
    def _preprocess_sentence(self, sentence: str) -> str:
        """
        Xoá tất cả dấu câu
        Viết thường tất cả ký tự
        """
        # string.punctuation = tất cả dấu câu
        # !"#$%&'()*+,-./:;<=>?@[\]^_{|}~
        translator = str.maketrans('', '', string.punctuation)
        sentence = sentence.lower()
        sentence = sentence.translate(translator) # Loại bỏ tất cả dấu câu
        return sentence
        
    def encode_sentence(self, sentence: str) -> torch.Tensor:
        sentence = self._preprocess_sentence(sentence)
        words = sentence.split()
        words = [self.bos] + words
        word_ids = [self.word2idx[word] for word in words]
        return torch.tensor(word_ids, dtype=torch.long)
    
    def encode_label(self, label: str) -> torch.Tensor:
        """
        Chuyển label dạng chữ (str) sang dạng số (tensor)
        """
        label_ids = self.label2idx[label]
        return torch.tensor(label_ids)
        
    def decode_label(self, label_ids: torch.Tensor) -> list:
        """
        Chuyển dạng số mà model dự đoán sang dạng chữ (str)
        """
        label_ids = label_ids.to_list()
        labels = [self.idx2label[label_id] for label_id in label_ids]
        return labels
    
class UIT_VSFC(Dataset):
    def __init__(
        self,
        filepath: str,
        vocab: Vocab,
        **kwargs: any
    ) -> None:
        super().__init__()
        
        self.vocab = vocab
        self.data = json.load(open(filepath))
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict:
        item = self.data[index]
        
        sentence: str = item["sentence"]
        label: str = item["topic"]
        
        input_ids = self.vocab.encode_sentence(sentence)
        label_ids = self.vocab.encode_label(label)
        
        return {
            "input_ids": input_ids,
            "label_ids": label_ids
        }
        
        