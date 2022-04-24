import torch.nn as nn
import torch
from model.backbone import Backbone
from model.mlp import MLP
from criterion import SetCriterion, HungarianMatcher

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, num_classes, num_queries, backbone = None, transformer = None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer or nn.Transformer(
            d_model = 256, #hidden_dim
            nhead = 8,
            num_encoder_layers = 6,
            num_decoder_layers = 6,
            dim_feedforward = 2048,
            dropout = 0.1,
            activation = "gelu", 
        )
        hidden_dim = self.transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.backbone = backbone or Backbone(
            name = "resnet34",
            train_backbone = True,
            pretrained = True,
            return_interm_layers = True,
            hidden_dim = hidden_dim
        )
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

    def forward(self, x):
        features, mask, pos_embed = self.backbone(x)
        src = self.input_proj(features).flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
 
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer(
            src = src + pos_embed,
            src_key_padding_mask = mask,
            memory_key_padding_mask = mask,
            tgt = tgt + query_embed
        )
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class.permute(1, 0, 2), 'pred_boxes': outputs_coord.permute(1, 0, 2)}
        return out


if __name__ == "__main__":
    num_classes = 5
    num_queries = 100
    model = DETR(num_classes, num_queries)
    matcher = HungarianMatcher(cost_class = 1, cost_bbox = 5, cost_giou = 2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)