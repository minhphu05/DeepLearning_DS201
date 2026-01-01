import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class ViOCD_Dataset(Dataset):
    def __init__(self, data_path: str, vocab, min_len: int = 3):
        """
        Args:
            data_path: Đường dẫn file json
            vocab: Object Vocabulary
            min_len: Độ dài tối thiểu của câu (tính bằng số từ). 
                     Các câu ngắn hơn sẽ bị loại bỏ để giảm nhiễu.
        """
        super().__init__()
        self.vocab = vocab
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        items = raw if isinstance(raw, list) else raw.values()
        
        self.data = []
        skipped_count = 0
        
        for item in items:
            if "review" in item and "domain" in item:
                review_text = item["review"]
                domain_label = str(item["domain"])
                
                processed_text = self.vocab._preprocess_sentence(review_text)
                if len(processed_text.split()) < min_len:
                    skipped_count += 1
                    continue
                # -----------------------------

                self.data.append({
                    "review": review_text,
                    "domain": domain_label
                })
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}. Skipped {skipped_count} short/empty samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_ids_list = self.vocab.encode_sentence(item["review"])
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        
        label_id = self.vocab.label2idx[item["domain"]]
        label_tensor = torch.tensor(label_id, dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "label_ids": label_tensor
        }

def collate_fn(items: list) -> dict:
    """
    Hàm gom batch.
    LƯU Ý: vocab.pad_idx mặc định là 0 trong vocab.py của bạn.
    Nếu bạn thay đổi thứ tự token trong vocab, hãy cập nhật value=... ở đây.
    """
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