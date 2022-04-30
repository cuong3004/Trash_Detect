import torch
from model.frozenbatchnorm import FrozenBatchNorm2d
from model.pe import PositionEmbeddingSine
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class Backbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 pretrained=True,
                 dilation=False,
                 norm_layer=None,
                 hidden_dim=256):
        super().__init__()
        self.backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained,
            norm_layer=norm_layer or FrozenBatchNorm2d
        )
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        N_steps = hidden_dim // 2
        self.position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    def forward(self, x, mask=None):
        mask = mask or torch.zeros_like(x[:, 0], dtype=torch.bool)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        m = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(x, m).to(x.dtype)

        return x, m, pos


def test_export():
    model = Backbone("resnet18")

    x = torch.rand(2, 3, 320, 320)
    _, m, pos = model(x)

    print(x.shape, m.shape, pos.shape)

    from export import export_tflite

    export_tflite(model, x, 'model.onnx',
                  input_names=["input"],
                  output_names=["output", "output1", "output2"])
