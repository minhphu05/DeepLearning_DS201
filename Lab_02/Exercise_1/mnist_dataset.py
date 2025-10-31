import torch
from torch.utils.data import Dataset
import idx2numpy 
import numpy as np

def collate_fn (items: list[dict]) -> dict[dict]:
    # items = [{
    #     # "image" : item["image"].expand_dims(axis=0),
    #     "image": np.expand_dims(item["image"], axis=0),
    #     "label": np.array(item["label"])
    # } for item in items]
    
    data = {
        "image": np.stack([item["image"] for item in items], axis=0),
        "label": np.stack([item["label"] for item in items], axis=0)
    }
    
    data = {
        "image": torch.tensor (data["image"]),
        "label": torch.tensor (data["label"])
    }
    
    return data

class Item:
    def __init__(self, image, label):
        self.image = image
        self.label = label

class MnistDataset(Dataset):
    def __init__(self, image_path: str, label_path: str):
        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)
        
        self._data = [{
            "image": np.array(image, dtype=np.float32),
            "label": label
        } for image, label in zip (images.tolist(), labels.tolist())
        ]
        
    def __len__(self) -> int:
        return len(self._data)
        
    def __getitem__(self, idx: int) -> dict:
        # print (self._data[idx]["image"].shape)
        return self._data[idx]
        
        