import torch.nn as nn
import torch
from model.backbone import Backbone
from model.mlp import MLP
from criterion import SetCriterion, HungarianMatcher


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, num_classes, num_queries, backbone=None, transformer=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer or nn.Transformer(
            d_model=256,  # hidden_dim
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            # activation = "gelu",
        )
        hidden_dim = self.transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.backbone = backbone or Backbone(
            name="resnet18",

            pretrained=True,

            hidden_dim=hidden_dim
        )
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=(1, 1))

    def forward(self, x):
        features, mask, pos_embed = self.backbone(x)
        src = self.input_proj(features).flatten(2).permute(2, 0, 1) # [64,2,256]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer(
            src=src + pos_embed,
            src_key_padding_mask=mask,
            memory_key_padding_mask=mask,
            tgt=tgt + query_embed
        )
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class.permute(1, 0, 2),
               'pred_boxes': outputs_coord.permute(1, 0, 2)}  # [2, 100, 6]
        return out


def test():
    num_classes = 4
    num_queries = 100
    model = DETR(num_classes, num_queries)
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)

    imgs = torch.randn((2, 3, 256, 256))
    outputs = model(imgs)

    targets = [
        {
            "boxes": torch.tensor(
                [[0.1, 0.2, 0.3, 0.6], [0.5, 0.6, 0.7, 0.8]] + [[0.0, 0.0, 1.0, 1.0] for i in range(40)]),
            "labels": torch.tensor([1, 1] + [num_classes for i in range(40)])
        },
        {
            "boxes": torch.tensor(
                [[0.1, 0.2, 0.3, 0.6], [0.5, 0.6, 0.7, 0.8]] + [[0.0, 0.0, 1.0, 1.0] for i in range(98)]),
            "labels": torch.tensor([1, 1] + [num_classes for i in range(98)])
        }
    ]

    losses = criterion(outputs, targets)
    print(sum(losses.values()), losses)


def test_real():
    from pathlib import Path
    from custom_data import CustomData
    import torchvision.transforms as T

    root_data = Path("data")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_train = CustomData(
        root=root_data / "train",
        annFile=root_data / 'train' / '_annotations.coco.json',
        transforms=transform)

    data_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=32,
        collate_fn=CustomData.collate_fn)

    imags, targets = next(iter(data_loader))

    num_classes = 5
    num_queries = 100
    model = DETR(num_classes, num_queries)
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)

    outputs = model(imags)

    # targets = [
    #     {
    #         "boxes":torch.tensor([[0.1,0.2,0.3,0.6], [0.5,0.6,0.7,0.8]]+[[0.0, 0.0, 1.0, 1.0] for i in range(40)]),
    #         "labels": torch.tensor([1, 1]+[num_classes for i in range(40)])
    #     },
    #     {
    #         "boxes":torch.tensor([[0.1,0.2,0.3,0.6], [0.5,0.6,0.7,0.8]]+[[0.0, 0.0, 1.0, 1.0] for i in range(98)]),
    #         "labels": torch.tensor([1, 1]+[num_classes for i in range(98)])
    #     }
    # ]

    losses = criterion(outputs, targets)
    print(sum(losses.values()), losses)


def test_export():
    num_classes = 4
    num_queries = 100
    model = DETR(num_classes, num_queries)

    x = torch.rand(2, 3, 320, 320)
    _, m, pos = model(x)

    print(x.shape, m.shape, pos.shape)

    from export import export_tflite

    export_tflite(model, x, 'model.onnx',
                  input_names=["input"],
                  output_names=["output", "output1"])

#
