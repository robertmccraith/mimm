from typing import Dict, List
import mlx.nn as nn
import mlx.core as mx
import mimm
from mimm.models.detection.detr.positional_embedding import PositionEmbeddingSine
from mimm.models.detection.detr.postprocess import PostProcess
from mimm.models.detection.detr.transformer import Transformer
from mimm.models.detection.detr.utils import (
    FrozenBatchNorm2d,
    NestedTensor,
    nested_tensor_from_tensor_list,
)


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        self.body = backbone
        self.num_channels = num_channels

    def __call__(self, tensor_list: NestedTensor):
        xs = self.body.features(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            interpolation = nn.Upsample(
                scale_factor=x.shape[-2] / m.shape[-2], mode="linear"
            )
            m = (
                interpolation(m[..., None].astype(mx.float32))
                .astype(mx.bool_)
                .squeeze(-1)
            )
            out[name] = NestedTensor(x, m)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        backbone = getattr(mimm.models.classification.resnet, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def __call__(self, tensor_list: NestedTensor):
        xs = self.layers[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self.layers[1](x))

        return out, pos


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def __call__(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, mx.array)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos[-1]
        )[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = nn.sigmoid(self.bbox_embed(hs))
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        return out


def detr_resnet50(num_classes=91, num_queries=100, hidden_dim=256):
    # Backbone
    position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone = Backbone("resnet50", True, False, False)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    transformer = Transformer(
        d_model=hidden_dim,
        dropout=0.1,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        normalize_before=False,
        return_intermediate_dec=True,
    )

    detr_model = DETR(
        model,
        transformer,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=True,
    )
    postprocessors = {"bbox": PostProcess()}

    return detr_model, postprocessors
