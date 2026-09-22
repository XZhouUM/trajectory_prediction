
from .mlp import MLP
from .transformer import Transformer

MODEL_REGISTRY = {
    "mlp": MLP,
    "transformer": Transformer,
}


def build_model(name: str, **kwargs):
    try:
        model_class = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {available}"
        ) from exc

    return model_class(**kwargs)