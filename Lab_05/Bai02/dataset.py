import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import os
from collections import Counter

def load_json_or_jsonl(filepath: str):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"File {filepath} có vẻ là JSON Lines. Đang chuyển sang chế độ đọc từng dòng...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue # Bỏ qua dòng lỗi
    return data

def extract_data(item: dict):
    tokens = item.get("tokens")
    if tokens is None:
        tokens = item.get("words")
    if tokens is None:
        tokens = item.get("syllables")
    
    tags = item.get("tags")
    if tags is None:
        tags = item.get("ner_tags")

    if tokens is None or tags is None:
        raise KeyError(f"Không tìm thấy key 'tokens'/'words' hoặc 'tags'. Keys hiện có: {list(item.keys())}")
        
    return tokens, tags

class Vocab:
    def __init__(self, train_filepath: str, min_freq: int = 1):
        word_counter = Counter()
        tag_set = set()

        print(f"Building vocab from: {train_filepath}")
        
        try:
            data = load_json_or_jsonl(train_filepath)
        except Exception as e:
            raise ValueError(f"Không thể đọc file {train_filepath}. Lỗi: {e}")

        # Kiểm tra dữ liệu rỗng
        if not data:
            raise ValueError("File dữ liệu rỗng!")

        for i, item in enumerate(data):
            try:
                tokens, tags = extract_data(item)
            except KeyError as e:
                if i == 0: raise e
                continue

            for toks in tokens:
                word_counter[toks.lower()] += 1
            for tag in tags:
                tag_set.add(tag)
        
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.pad_tag = "<pad_tag>"

        self.word2idx = {
            self.pad: 0,
            self.unk: 1,
        }

        for w, c in word_counter.items():
            if c >= min_freq:
                self.word2idx[w] = len(self.word2idx)
        
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.tag2idx = {
            tag: i for i, tag in enumerate(sorted(list(tag_set)))
        }
        self.tag2idx[self.pad_tag] = -100 
        self.idx2tag = {i: t for t, i in self.tag2idx.items()}
        
        print(f"Vocab size: {len(self.word2idx)}")
        print(f"Num tags: {len(self.tag2idx)-1}") # Trừ pad_tag
    
    def encode_tokens(self, tokens: list[str]) -> torch.Tensor:
        ids = []
        for token in tokens:
            token = token.lower()
            ids.append(self.word2idx.get(token, self.word2idx[self.unk]))
        return torch.tensor(ids, dtype=torch.long)
    
    def encode_tags(self, tags: list[str]) -> torch.Tensor:
        ids = [self.tag2idx.get(tag, -100) for tag in tags]
        return torch.tensor(ids, dtype=torch.long)
    
    def decode_tokens(self, ids: list[int]) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [self.idx2word.get(i, self.unk) for i in ids]

    def decode_tags(self, ids: list[int]) -> list[str]:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [self.idx2tag.get(i, "PAD") for i in ids]

    @property
    def n_tags(self) -> int:
        return len([i for i in self.tag2idx.values() if i != -100])
    
    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)


class PhoNER_COVID19(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        
        print(f"Loading data from: {path}")
        self.data = load_json_or_jsonl(path)
    
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

def collate_fn(items: list[dict]) -> dict:
    input_ids = [item["input_ids"] for item in items]
    tags_ids = [item["tags_ids"] for item in items]
    
    lengths = torch.tensor([len(seq) for seq in input_ids], dtype=torch.long)
    
    padded_input = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_tags = pad_sequence(tags_ids, batch_first=True, padding_value=-100)

    return {
        "input_ids": padded_input,
        "tags_ids": padded_tags,
        "lengths": lengths
    }