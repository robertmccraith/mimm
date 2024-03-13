from typing import Dict, List
from mlx.utils import tree_flatten
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import torch
from torch.hub import download_url_to_file, get_dir


def tree_unflatten(tree):
    """Recreate a python tree from its flat representation.

    .. code-block:: python

        from mlx.utils import tree_unflatten

        d = tree_unflatten([("hello.world", 42)])
        print(d)
        # {"hello": {"world": 42}}

    Args:
        tree (List[Tuple[str, Any]]): The flat representation of a python tree.
                                      For instance as returned by :meth:`tree_flatten`.

    Returns:
        A python tree.
    """
    if len(tree) == 1 and tree[0][0] == "":
        return tree[0][1]

    try:
        int(tree[0][0].split(".", maxsplit=1)[0])
        is_list = True
    except ValueError:
        is_list = False

    # collect children
    children = {}
    for key, value in tree:
        current_idx, *next_idx = key.split(".", maxsplit=1)
        next_idx = "" if not next_idx else next_idx[0]
        if current_idx not in children:
            children[current_idx] = []
        children[current_idx].append((next_idx, value))

    # recursively map them to the original container
    if is_list:
        keys = sorted((int(idx), idx) for idx in children.keys())
        layers = []
        for i, k in keys:
            while i > len(layers):
                layers.append({})
            layers.append(tree_unflatten(children[k]))
        if all(a.isdigit() for a in children.keys()):
            return {"layers": layers}
        return layers
    else:
        d = {k: tree_unflatten(v) for k, v in children.items()}
        return d


def apply(dst, parameters):
    if isinstance(parameters, dict):
        for k in parameters:
            if k in dst:
                current_value = dst[k]
                new_value = parameters[k]
                if isinstance(current_value, mx.array):
                    dst[k] = new_value
                elif isinstance(current_value, nn.Module):
                    apply(current_value, new_value)
                elif isinstance(current_value, (dict, list)):
                    apply(current_value, new_value)
    elif isinstance(parameters, list):
        for i in range(len(parameters)):
            current_value = dst[i]
            new_value = parameters[i]
            if isinstance(current_value, mx.array):
                dst[i] = new_value
            elif isinstance(current_value, nn.Module):
                apply(current_value, new_value)
            elif isinstance(current_value, (dict, list)):
                apply(current_value, new_value)


def get_pytorch_weights(weights_url: str) -> Dict:
    hub_dir = get_dir()
    model_dir = Path(hub_dir) / "checkpoints"

    filename = Path(weights_url).name
    cached_file = model_dir / filename
    if not cached_file.exists():
        download_url_to_file(weights_url, cached_file)
    return cached_file


def load_pytorch_weights(
    model,
    weights: Path,
    conv_layers: List[str],
    padding_layers=[],
    layer_name_changes=None,
    layer_modify=None,
):
    weights = torch.load(weights)
    weights = tree_flatten(weights)
    w2 = []
    for k, v in weights:
        v = mx.array(v.detach().cpu().numpy())
        if any(conv_layer in k for conv_layer in conv_layers) and len(v.shape) == 4:
            v = v.transpose(0, 2, 3, 1)
        if layer_name_changes:
            kvs = layer_name_changes(k, v)
            for k, v in kvs:
                w2.append((k, v))
        else:
            w2.append((k, v))
    w2 += [(name, {}) for name in padding_layers]
    ws = tree_unflatten(w2)
    import numpy as np

    pre_weights = {
        k: np.array(v.tolist())
        for k, v in tree_flatten(model.children())
        if isinstance(v, mx.array)
    }
    if layer_modify:
        layer_modify(ws)
    apply(model.children(), ws)
    post_weights = {
        k: np.array(v.tolist())
        for k, v in tree_flatten(model.children())
        if isinstance(v, mx.array)
    }
    for k, v in pre_weights.items():
        if np.linalg.norm(v - post_weights[k]) < 0.1:
            print(k)
