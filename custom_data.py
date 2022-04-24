from torchvision.datasets.coco import CocoDetection
from pathlib import Path
import torch 
from typing import Any, Callable, Optional, Tuple, List


root_data = Path("data")

data_train = CocoDetection(
                root=root_data/"train",
                annFile=root_data/'train'/'_annotations.coco.json')

class CustomData(CocoDetection):
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        target = self._preprocess_target(target)

        return {"image": image, "target": target}
    
    def _preprocess_target(self, target):
        target_new = {}
        boxxes = []
        categories_id = []
        for i in range(len(target)):
            boxxes.append(torch.tensor(target[i]["bbox"]))
            categories_id.append(torch.tensor(target[i]["category_id"]))
        
        boxxes = torch.stack(boxxes)
        categories_id = torch.stack(categories_id)

        target_new["boxxes"] = boxxes
        target_new["categories_id"] = categories_id

        return target_new



