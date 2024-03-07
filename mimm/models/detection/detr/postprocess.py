import mlx.nn as nn
import mlx.core as mx


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.transpose(2, 0, 1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return mx.stack(b, axis=-1)


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __call__(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = nn.softmax(out_logits, -1)
        scores = prob[..., :-1].max(-1)
        labels = prob[..., :-1].argmax(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.transpose(1, 0)
        scale_fct = mx.stack([img_w, img_h, img_w, img_h], axis=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": label, "boxes": b}
            for s, label, b in zip(scores, labels, boxes)
        ]

        return results
