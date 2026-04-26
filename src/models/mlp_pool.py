"""Mean-pool MLP router for ASR model selection.

This is a deliberately simple baseline that mean-pools each expert's
frame-level encoder outputs to a single vector (respecting the
attention mask), concatenates the resulting per-expert vectors, and
feeds the concatenation through a small MLP to predict a probability
distribution over the experts.

The forward signature matches :class:`src.models.selector.ASRModelSelector`
so the same training script and dataset can be used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPoolSelector(nn.Module):
    """Mean-pool baseline for per-clip ASR model selection."""

    def __init__(
        self,
        model_dims: dict[str, int],
        model_names: list[str],
        d_hidden: int = 1024,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Args:
            model_dims: Mapping of model name to encoder hidden dimension.
            model_names: Ordered list of ASR model names. The output
                logits follow this order.
            d_hidden: Width of each hidden layer in the MLP.
            n_layers: Number of hidden layers in the MLP. Must be >= 1.
            dropout: Dropout probability between hidden layers.
        """
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.model_names = model_names
        self.n_models = len(model_names)
        total_in = sum(model_dims[name] for name in model_names)

        layers: list[nn.Module] = []
        prev = total_in
        for _ in range(n_layers):
            layers += [
                nn.Linear(prev, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = d_hidden
        layers.append(nn.Linear(prev, self.n_models))
        self.classifier = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: dict[str, torch.Tensor],
        attention_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Dict mapping model name to encoder outputs of
                shape ``(B, T_k, D_k)``.
            attention_masks: Dict mapping model name to boolean masks of
                shape ``(B, T_k)``, ``True`` where valid.

        Returns:
            Tensor of shape ``(B, n_models)`` with row-stochastic
            routing probabilities.
        """
        pooled = []
        for name in self.model_names:
            x = hidden_states[name]
            mask = attention_masks[name].unsqueeze(-1).to(x.dtype)
            summed = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled.append(summed / denom)
        feat = torch.cat(pooled, dim=-1)
        logits = self.classifier(feat)
        return F.softmax(logits, dim=-1)

    def count_parameters(self) -> dict:
        """Return parameter count by component, including total.

        Mirrors :meth:`ASRModelSelector.count_parameters` so external
        logging code does not need to special-case the architecture.
        """
        return {
            "classifier": sum(p.numel() for p in self.classifier.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
