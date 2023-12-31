from pathlib import Path
from typing import List
from mlx.utils import tree_flatten
import mlx.core as mx


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


def load_pytorch_weights(model, weights_path: Path, conv_layers: List[str]):
    import torch

    weights = torch.load(weights_path, map_location="cpu")
    weights = tree_flatten(weights)
    w2 = []
    for k, v in weights:
        v = mx.array(v.detach().cpu().numpy())
        if any(conv_layer in k for conv_layer in conv_layers) and len(v.shape) == 4:
            v = v.transpose(0, 2, 3, 1)
        w2.append((k, v))
    ws = tree_unflatten(w2)
    model.update(ws)
