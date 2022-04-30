# from torchmetrics.functional import accuracy 
# from torchmetrics import Precision, Recall, F1Score, Accuracy
from model.detr import DETR
from custom_data import CustomData
from criterion import SetCriterion, HungarianMatcher
import pytorch_lightning as pl
import torch.nn as nn
import torch
from utils.boxOps import BoxUtils
import copy
from pathlib import Path
from custom_data import CustomData
import torchvision.transforms as T
from pycocotools.cocoeval import COCOeval


class LitClassification(pl.LightningModule):
    def __init__(self,
                 image_size=(320, 320),
                 num_classes=4,
                 hidden_dim=256,
                 num_queries=100,
                 nheads=8,
                 dropout=0.1,
                 dim_feedforward=2048,
                 enc_layers=6,
                 dec_layers=6,
                 # pre_norm = False,
                 # deep_supervision=True,
                 ce_weight=1.0,
                 bbox_weight=5.0,
                 giou_weight=2.0,
                 train_coco=None,
                 valid_coco=None,
                 test_coco=None,
                 ):
        super().__init__()

        transformer = nn.Transformer(
            d_model=hidden_dim,  # hidden_dim
            nhead=nheads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
        )
        self.detr = DETR(num_classes=num_classes, num_queries=num_queries, transformer=transformer)
        matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
        weight_dict = {'loss_ce': ce_weight, 'loss_bbox': bbox_weight, 'loss_giou': giou_weight}
        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1,
                                      losses=losses)

        self.train_coco = train_coco
        self.valid_coco = valid_coco
        self.test_coco = test_coco
        self.annType = 'bbox'

    def shared_step(self, batch, mode, **kwargs):
        images, targets = batch
        out = self.detr(images)

        loss = self.criterion(out, targets)
        loss = sum(loss.values())

        self.log(f'{mode}_loss', loss.item(), **kwargs)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.detr.parameters(), lr=0.0001)

    def training_step(self, train_batch, batch_idx):
        loss = self.shared_step(train_batch, "train", on_step=False, on_epoch=True)
        return loss

    # def validation_step(self, val_batch, batch_idx):
    #     self.shared_step(val_batch, "valid", on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    # def predict_step(self, batch, batch_idx):
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        out = self.detr(images)
        return out["pred_logits"], out["pred_boxes"]

    def _out2json(self, out):
        pred_logits = out["pred_logits"]
        pred_boxes = out["pred_boxes"]

    def on_predict_epoch_end(self, outputs):
        pred_logits = [output[0] for output in outputs]
        pred_boxes = [output[1] for output in outputs]
        assert len(pred_logits) == len(pred_boxes)

        pred_logits = torch.cat(pred_logits)
        pred_boxes = torch.cat(pred_boxes)

        pred_probas = pred_logits.softmax(-1)[:, :, :-1]
        keep = pred_probas.max(-1).values > 0.5

        bboxes_scaled = BoxUtils.rescale_bboxes(pred_boxes[:, keep], (320,320))
        pred_probas = pred_probas[keep]

        # idxes = torch.arange(pred_logits.shape[0], type=torch.int64)
        pred_result = []
        for idx, (boxes, prob) in enumerate(zip(bboxes_scaled, pred_probas)):
            for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
                cl = p.argmax()
                score = p[cl]
                target_pred = {"image_id":idx,
                               "category_id":cl+1,
                               "bbox":[xmin, ymin, xmax, ymax],
                               "score":score}
                pred_result.append(target_pred)

        cocoGt = copy.deepcopy(self.test_coco)
        cocoDt = cocoGt.loadRes(pred_result)

        cocoEval = COCOeval(cocoGt,cocoDt,self.annType)
        # cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


        # return
        # target_pred = {}



root_data = Path("data")

transform_train = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_valid = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data_train = CustomData(
    root=root_data / "train",
    annFile=root_data / 'train' / '_annotations.coco.json',
    transforms=transform_train)

data_valid = CustomData(
    root=root_data / "test",
    annFile=root_data / 'test' / '_annotations.coco.json',
    transforms=transform_train)

train_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size=32,
    collate_fn=CustomData.collate_fn)

valid_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size=32,
    collate_fn=CustomData.collate_fn)

# trainer = pl.Trainer(max_epochs=20)
# trainer.fit(LitClassification(), train_loader,)
# trainer.predict(valid_loader)

trainer = pl.Trainer(gpus=1, max_epochs=1)
lit = LitClassification(
    train_coco=data_train.coco,
    valid_coco=data_valid.coco,)
trainer.fit(lit, train_loader, valid_loader)

# x = torch.randn(2, 3, 320, 320, requires_grad=True)
# lit.detr.eval()
# torch_out = lit.detr(x)

# Export the model
# torch.onnx.export(lit.detr,  # model being run
#                   x,  # model input (or a tuple for multiple inputs)
#                   "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
#                   opset_version=12,  # the ONNX version to export the model to
#                   input_names=['input'],  # the model's input names
#                   output_names=['output', 'output2'],  # the model's output names
#                   )
