import typing
import warnings

import torch
from transformers import AutoTokenizer

if typing.TYPE_CHECKING:
    from .vector import SteeringVector, SteeringModel


def get_activation_score(
    input_text: str,
    model: "SteeringModel",
    control_vector: "SteeringVector",
    layer_index=None,  # can be int or list of ints
    scoring_method: str = "mean",  # 'mean', 'final_token', 'max_token', or 'median_token'
) -> float:
    """
    Compute the activation score for input_text by projecting hidden states onto
    the control_vector direction(s) at the specified layer(s).

    Scoring methods:
        - 'mean': Average dot products over all tokens.
        - 'final_token': Dot product of the final token.
        - 'max_token': Maximum dot product among tokens.
        - 'median_token': Median dot product among tokens.

    Args:
        input_text (str): The input string to evaluate.
        model (SteeringModel): The model to use for computing activations.
        control_vector (SteeringVector): Contains direction(s) keyed by layer index.
        layer_index (int or list[int], optional): Layer(s) to use. Defaults to last in model.layer_ids.
        scoring_method (str): Scoring method to use.

    Returns:
        float: Averaged activation score across selected layers.
        int: Number of tokens in the input.
        list: Unaggregated dot product scores for each layer.
    """
    # 1) Reset the model to ensure no control is applied.
    model.reset()

    # 2) Determine the layer(s) to use.
    if layer_index is None:
        if not model.layer_ids:
            raise ValueError("No controlled layers set on this model!")
        layer_index = model.layer_ids[-1]

    # If a single int is provided, wrap it in a list for unified processing.
    if not isinstance(layer_index, list):
        layers_to_use = [layer_index]
    else:
        layers_to_use = layer_index

    # 3) Prepare a container to store hidden states for each requested layer.
    hook_states = {}

    # 4) Define and register a hook function for each layer.
    def get_hook_fn(key):
        def hook_fn(module, inp, out):
            # If out is a tuple (hidden, present, ...), take the first element.
            if isinstance(out, tuple):
                hook_states[key] = out[0]
            else:
                hook_states[key] = out

        return hook_fn

    # 5) Retrieve the list of layers from the model.
    def model_layer_list(m):
        if hasattr(m, "model"):
            return m.model.layers
        elif hasattr(m, "transformer"):
            return m.transformer.h
        else:
            raise ValueError("Cannot locate layers for this model type")

    layers = model_layer_list(model.model)

    # 6) For each provided layer index, compute its actual index and register the hook.
    hooks = []
    for li in layers_to_use:
        real_layer_idx = li if li >= 0 else len(layers) + li
        hook_handle = layers[real_layer_idx].register_forward_hook(get_hook_fn(li))
        hooks.append(hook_handle)

    # 7) Build a tokenizer from the model name.
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # 8) Encode the input text and perform a forward pass.
    encoded = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    with torch.no_grad():
        _ = model.model(input_ids)

    # 9) Remove hooks to clean up.
    for hook in hooks:
        hook.remove()

    # 10) For each layer, compute the activation score using the chosen scoring method.
    scores = []
    unaggregated_scores = []
    for li in layers_to_use:
        if li not in hook_states:
            raise RuntimeError(
                f"Did not capture hidden states for layer {li} in the forward pass!"
            )
        # Extract hidden states for the single batch: shape [seq_len, hidden_dim]
        hidden_states = hook_states[li][0]
        # Retrieve the corresponding direction from the control_vector.
        if li not in control_vector.directions:
            raise ValueError(f"No direction for layer {li} in control_vector!")
        direction_np = control_vector.directions[li]
        direction = torch.tensor(
            direction_np, device=model.device, dtype=model.model.dtype
        )

        # Compute dot products for all tokens: shape [seq_len]
        dot_vals = hidden_states @ direction
        token_length = dot_vals.shape[0]
        # Determine score based on the scoring_method.
        if scoring_method == "mean":
            # Average over all tokens.
            score_tensor = dot_vals.mean()
        elif scoring_method == "final_token":
            # Use only the final token.
            score_tensor = dot_vals[-1]
        elif scoring_method == "max_token":
            # Use the maximum token's dot product.
            score_tensor = dot_vals.max()
        elif scoring_method == "median_token":
            # Use the median token's dot product.
            score_tensor = dot_vals.median()
        else:
            raise ValueError(f"Unknown scoring_method: {scoring_method}")
        unaggregated_score = dot_vals.cpu().detach().numpy()
        unaggregated_scores.append(unaggregated_score)
        scores.append(score_tensor.item())

    # 11) Average the scores across the selected layers.
    avg_score = sum(scores) / len(scores)
    return avg_score, token_length, unaggregated_scores