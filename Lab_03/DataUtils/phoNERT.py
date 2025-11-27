import os
import json
import string
from collections import Counter
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

"""
B-XXX: Beginning of Entity
I-XXX: Inside of Entity
O    : Outside of Entity
"""

def collate_fn(items: list[dict]) -> dict:
    """
    - Gom các input_ids lại
    - Gom các tags_ids lại
    - Padding cho tất cả câu có độ dài khác nhau
    - Trả ra tensor batch để đưa vào model
    """
    # Lấy các input và label từ từng item
    input_ids = [item["input_ids"] for item in items]
    tags_ids = [item["tags_ids"] for item in items]
    
    # Lưu lại độ dài thực tế của từng câu
    lengths = torch.tensor([len(seq) for seq in input_ids], dtype=torch.long)
    
    """
    Example:
        [3, 5, 7, 2]
        [4, 1]
        [9, 8, 3]    
    -->
        [
            [3, 5, 7, 2],
            [4, 1, 0, 0],
            [9, 8, 3, 0]
        ]
    """
    padded_inputs = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )
    padded_tags = pad_sequence(
        tags_ids,
        batch_first=True,
        padding_value=-100
    )
    return {
        "input_ids": padded_inputs,
        "tags_ids": padded_tags,
        "lengths": lengths
    }

def read_json_or_jsonl(filepath: str) -> list:
    """
    Đọc file json hoặc jsonl
    """
    data = []
    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"File {filepath} có vẻ là JSON Lines. Đang chuyển sang chế độ đọc từng dòng...")
        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue # Bỏ qua lỗi
    print("Đã load xong dữ liệu")
    return data

def extract_data(item: dict):
    tokens: list[str] = item.get("tokens")
    if tokens is None:
        tokens = item.get("words")
    if tokens is None:
        tokens = item.get("syllables")
    
    tags = item.get("tags")
    if tags is None:
        tags: list[str] = item.get("ner_tags")
        
    if tokens is None or tags is None:
        raise ValueError(f"Không thể tìm thấy key 'tokens'/'words'/'syllables' hoặc 'tags'/'ner_tags'.\nKey hiện có {list(item.keys())}")
    return tokens, tags
                        
class Vocab:
    def __init__(
        self,
        filepath: str
    ):
        """
            filepath: Chỉ đọc dữ liệu từ tập train để tạo Vocab
        """
        # {phần_tử: số_lần_xuất_hiện}
        word_counter = Counter()
        tag_set = set()
        
        print(f"Building vocab from {filepath}")
        try:
           data = read_json_or_jsonl(filepath)
        except Exception as e:
            raise ValueError(f"Không thể đọc file {filepath}.\nLỗi {e}")
        
        if not data:
            raise ValueError("File dữ liệu rỗng")
        
        for idx, item in enumerate(data):
            try:
                tokens, tags = extract_data(item)
            except KeyError as e:
                if idx == 0: 
                    raise e
                continue
            
            for token in tokens:
                word_counter[token.lower()] += 1
            for tag in tags:
                tag_set.add(tag)
        
        # Padding Token: Dùng để chèn vào cuối câu sao cho tất cả câu trong batch cùng độ dài.
        self.pad = "<pad>"
        # Unknown Token: Dùng cho những từ không có trong vocab.
        self.unk = "<unk>"
        # Padding Tag (NER or POSTagging): 
        self.pad_tag = "<pad_tag>"
        
        self.word2idx = {
            self.pad: 0,
            self.unk: 1
        }
        
        for word, count in word_counter.items():
            if count >= 1:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {
            idx: word for word, idx in self.word2idx.items()
        }
        
        self.tag2idx = {
            tag: idx for idx, tag in enumerate(sorted(list(tag_set)))
        }
        self.tag2idx[self.pad_tag] = -100
        self.idx2tag = {
            idx: tag for tag, idx in self.tag2idx.items()
        }
        
        print(f"Vocab size: {len(self.word2idx)}")
        print(f"Num tags: {len(self.tag2idx)-1}") # Trừ pad_tag
    
    @property
    def num_tags(self) -> int:
        return len([i for i in self.tag2idx.values() if i != -100])
    
    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)
    
    def encode_tokens(
        self,
        tokens: list[str]
    ) -> torch.Tensor:
        """
        """
        ids = []
        for token in tokens:
            token = token.lower()
            ids.append(
                # dict.get(key, default) --> Nếu không có key, trả về default
                self.word2idx.get(token, self.word2idx[self.unk])
            )
        return torch.tensor(ids, dtype=torch.long)
    
    def encode_tags(
        self,
        tags: list[str]
    ) -> torch.Tensor:
        """
        """
        ids = [self.tag2idx.get(tag, -100) for tag in tags]
        return torch.tensor(ids, dtype=torch.long)
            
    def decode_tokens(
        self,
        ids: list[int]
    ) -> list[str]:
        # isinstance(object, type): Check datatype of a variable
        if isinstance(ids, torch.Tensor):
           ids = ids.tolist()
        return [self.idx2word.get(id, self.unk) for id in ids]    
    
    def decode_tags(
        self,
        ids: list[int]
    ) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [self.idx2tag.get(id, "PAD") for id in ids]
    
class phoNERT(Dataset):
    def __init__(
        self,
        filepath: str,
        vocab: Vocab,
        **kwargs: any
    ) -> None:
        super().__init__()
        
        self.vocab = vocab
        print(f"Loading data from {filepath}")
        self.data = read_json_or_jsonl(filepath)
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict:
        item = self.data[index]
        
        tokens, tags = extract_data(item)
        
        input_ids = self.vocab.encode_tokens(tokens)
        tags_ids = self.vocab.encode_tags(tags)
        
        return {
            "input_ids": input_ids,
            "tags_ids": tags_ids
        }
        
