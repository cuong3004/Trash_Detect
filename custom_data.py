from torchvision.datasets.coco import CocoDetection
from pathlib import Path
import torch 
from typing import Any, Callable, Optional, Tuple, List
from utils.boxOps import BoxUtils



class CustomData(CocoDetection):
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image = self.transforms(image)
        
        target = self._preprocess_target(image, target)

        return image, target
    
    def _preprocess_target(self, image, target):
        target_new = {}
        boxxes = []
        categories_id = []
        for i in range(len(target)):
            bbox = torch.tensor(target[i]["bbox"])
            bbox = BoxUtils.downscale_bboxes(bbox, image.shape[-2:])
            category = torch.tensor(target[i]["category_id"]) - 1

            boxxes.append(bbox)
            categories_id.append(category)
        
        boxxes = torch.stack(boxxes)
        categories_id = torch.stack(categories_id)

        target_new["boxes"] = boxxes
        target_new["labels"] = categories_id

        return target_new

    @staticmethod
    def collate_fn(batch):
        images, targets = [], []
        for image, target in batch:
            images.append(image)
            targets.append(target)

        images = torch.stack(images)
        return images, targets

def test():
    import  torchvision.transforms as T
    
    root_data = Path("data")

    transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    data_train = CustomData(
                    root=root_data/"train",
                    annFile=root_data/'train'/'_annotations.coco.json',
                    transforms=transform)

    
    data_loader = torch.utils.data.DataLoader(
        data_train, 
        batch_size=32, 
        collate_fn=CustomData.collate_fn)




    print(next(iter(data_loader)))


    print("Okela")


