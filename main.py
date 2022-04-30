from pe import PositionEmbeddingSine, Joiner
from mobile_vit import mobilevit_s
import torch.nn as nn
import torch


class Detr(nn.Module):
    """
    Implement Detr
    """

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
                 pre_norm=False,
                 deep_supervision=True,
                 giou_weight=2.0,
                 l1_weight=5.0,
                 no_object_weight=1.0,
                 ):
        super().__init__()

        N_steps = hidden_dim // 2
        vit_backbone = mobilevit_s()
        backbone = Joiner(vit_backbone, PositionEmbeddingSine(N_steps, normalize=True))

        transformer = nn.Transformer(
            hidden_dim=hidden_dim,
            nhead=nheads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )

        self.detr = DETR(
            backbone, transformer, num_classes=self.num_classes, num_queries=num_queries, aux_loss=deep_supervision
        )

        # self.device = torch.device(cfg.MODEL.DEVICE)

        # self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        # self.mask_on = cfg.MODEL.MASK_ON
        # hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        # num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        # # Transformer parameters:
        # nheads = cfg.MODEL.DETR.NHEADS
        # dropout = cfg.MODEL.DETR.DROPOUT
        # dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        # enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        # dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        # pre_norm = cfg.MODEL.DETR.PRE_NORM

        # # Loss parameters:
        # giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        # l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        # deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        # no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT

        # N_steps = hidden_dim // 2
        # d2_backbone = MaskedBackbone(cfg)
        # backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        # backbone.num_channels = d2_backbone.num_channels

        # transformer = Transformer(
        #     d_model=hidden_dim,
        #     dropout=dropout,
        #     nhead=nheads,
        #     dim_feedforward=dim_feedforward,
        #     num_encoder_layers=enc_layers,
        #     num_decoder_layers=dec_layers,
        #     normalize_before=pre_norm,
        #     return_intermediate_dec=deep_supervision,
        # )

        # self.detr = DETR(
        #     backbone, transformer, num_classes=self.num_classes, num_queries=num_queries, aux_loss=deep_supervision
        # )
        # if self.mask_on:
        #     frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
        #     if frozen_weights != '':
        #         print("LOAD pre-trained weights")
        #         weight = torch.load(frozen_weights, map_location=lambda storage, loc: storage)['model']
        #         new_weight = {}
        #         for k, v in weight.items():
        #             if 'detr.' in k:
        #                 new_weight[k.replace('detr.', '')] = v
        #             else:
        #                 print(f"Skipping loading weight {k} from frozen model")
        #         del weight
        #         self.detr.load_state_dict(new_weight)
        #         del new_weight
        #     self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ''))
        #     self.seg_postprocess = PostProcessSegm

        # self.detr.to(self.device)

        # # building criterion
        # matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        # weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        # weight_dict["loss_giou"] = giou_weight
        # if deep_supervision:
        #     aux_weight_dict = {}
        #     for i in range(dec_layers - 1):
        #         aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        #     weight_dict.update(aux_weight_dict)
        # losses = ["labels", "boxes", "cardinality"]
        # if self.mask_on:
        #     losses += ["masks"]
        # self.criterion = SetCriterion(
        #     self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
        # )
        # self.criterion.to(self.device)

        # pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        # pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        # self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        # self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        output = self.detr(images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({'masks': gt_masks})
        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
        )):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = mobilevit_s()
        # del self.backbone.fc

        # create conversion layer
        # self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
        )

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        # x = self.backbone.conv1(inputs)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)

        # x = self.backbone.layer1(x)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        h = self.backbone(inputs)

        # convert from 2048 to 256 feature planes for the transformer
        # h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        # os_embed.flatten(2).permute(2, 0, 1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
