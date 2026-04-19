"""
Hierarchical Transformer for ASR Model Selection.

Architecture:
    Stage 1 (Per-Model Temporal Encoder):
        - Linear projection from model-specific D_k to shared D_model
        - Sinusoidal positional encoding
        - Learnable [CLS] token per model (or shared + model embedding)
        - Shared-weight transformer encoder with model-identity embeddings
        → Produces one summary vector h_k per model

    Optional Cross-Attention Bridge:
        - Each model's CLS token cross-attends to other models' full sequences
        → Enables fine-grained cross-model temporal interaction

    Stage 2 (Cross-Model Fusion):
        - Small transformer over {h_1, h_2, h_3, h_manual, [CLS]}
        - Global CLS → MLP → 3-way softmax (model selection probabilities)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    No learnable parameters — generalizes to unseen sequence lengths.
    """

    def __init__(self, d_model: int, max_len: int = 4000, dropout: float = 0.1):
        """Args:
            d_model: Model embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Input tensor of shape (B, T, D).

        Returns:
            Positionally encoded tensor of shape (B, T, D).
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PerModelProjection(nn.Module):
    """Separate linear projection for each base model's encoder outputs.

    Maps from model-specific D_k to shared D_model.
    """

    def __init__(self, model_dims: dict[str, int], d_model: int):
        """Args:
            model_dims: Mapping of model name to feature dimension,
                e.g. {"hubert": 1024, "whisper": 1280, "wav2vec2": 768}.
            d_model: Shared transformer dimension.
        """
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, d_model)
            for name, dim in model_dims.items()
        })

    def forward(self, model_name: str, x: torch.Tensor) -> torch.Tensor:
        """Args:
            model_name: Name of the source ASR model.
            x: Input tensor of shape (B, T, D_k).

        Returns:
            Projected tensor of shape (B, T, D_model).
        """
        return self.projections[model_name](x)


class CrossAttentionBridge(nn.Module):
    """Optional cross-attention bridge.

    Each model's CLS token attends to the full temporal sequences of the
    OTHER models. Cheap (1 query vs ~T keys) but expressive: lets each
    model summary be informed by fine-grained temporal details from
    competing models.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        """Args:
            d_model: Model embedding dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cls_token: torch.Tensor,
        other_sequences: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            cls_token: Query CLS token of shape (B, 1, D).
            other_sequences: Key/value sequences from other models, (B, T_total, D).
            key_padding_mask: Boolean mask of shape (B, T_total), True where padded.

        Returns:
            Updated CLS token of shape (B, 1, D).
        """
        attn_out, _ = self.cross_attn(
            query=cls_token,
            key=other_sequences,
            value=other_sequences,
            key_padding_mask=key_padding_mask,
        )
        cls_token = self.norm(cls_token + self.dropout(attn_out))
        return cls_token


class ASRModelSelector(nn.Module):
    """Hierarchical Transformer for selecting the best ASR model per audio clip."""

    def __init__(
        self,
        model_dims: dict[str, int],
        model_names: list[str],
        d_model: int = 256,
        n_heads: int = 4,
        stage1_layers: int = 2,
        stage2_layers: int = 1,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        use_cross_attention_bridge: bool = True,
        share_stage1_weights: bool = True,
        max_seq_len: int = 2500,
    ):
        """Args:
            model_dims: Mapping of model name to encoder hidden dimension.
            model_names: Ordered list of ASR model names.
            d_model: Shared transformer hidden dimension.
            n_heads: Number of attention heads in all transformers.
            stage1_layers: Number of transformer layers in Stage 1.
            stage2_layers: Number of transformer layers in Stage 2.
            ffn_dim: Feed-forward dimension in transformer layers.
            dropout: Dropout probability.
            use_cross_attention_bridge: Whether to use the cross-attention bridge.
            share_stage1_weights: If True, all models share Stage 1 weights.
            max_seq_len: Maximum sequence length for positional encoding.
        """
        super().__init__()

        self.model_names = model_names
        self.n_models = len(model_names)
        self.d_model = d_model
        self.use_cross_attention_bridge = use_cross_attention_bridge
        self.share_stage1_weights = share_stage1_weights

        self.projection = PerModelProjection(model_dims, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )
        self.model_embeddings = nn.Embedding(self.n_models, d_model)
        self.cls_tokens = nn.Parameter(torch.randn(self.n_models, 1, d_model) * 0.02)

        if share_stage1_weights:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.stage1_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=stage1_layers
            )
        else:
            self.stage1_encoders = nn.ModuleDict()
            for name in model_names:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.stage1_encoders[name] = nn.TransformerEncoder(
                    encoder_layer, num_layers=stage1_layers
                )

        if use_cross_attention_bridge:
            self.cross_attention_bridges = nn.ModuleDict({
                name: CrossAttentionBridge(d_model, n_heads, dropout)
                for name in model_names
            })

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.stage2_encoder = nn.TransformerEncoder(
            fusion_layer, num_layers=stage2_layers
        )

        self.global_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.n_models),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _run_stage1(
        self,
        model_idx: int,
        model_name: str,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Stage 1 for a single model.

        Args:
            model_idx: Integer index of the model in self.model_names.
            model_name: Name of the model.
            hidden_states: Encoder outputs of shape (B, T, D_k).
            attention_mask: Boolean mask of shape (B, T), True where valid.

        Returns:
            cls_output: CLS summary of shape (B, 1, D_model).
            full_sequence: Full Stage 1 output of shape (B, T+1, D_model).
            padding_mask: Boolean mask of shape (B, T+1), True where padded.
        """
        B = hidden_states.size(0)

        x = self.projection(model_name, hidden_states)
        x = self.pos_encoder(x)

        model_emb = self.model_embeddings(
            torch.tensor(model_idx, device=x.device)
        )
        x = x + model_emb.unsqueeze(0).unsqueeze(0)

        cls = self.cls_tokens[model_idx].unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        extended_mask = torch.cat([cls_mask, attention_mask], dim=1)
        padding_mask = ~extended_mask

        if self.share_stage1_weights:
            x = self.stage1_encoder(x, src_key_padding_mask=padding_mask)
        else:
            x = self.stage1_encoders[model_name](x, src_key_padding_mask=padding_mask)

        cls_output = x[:, 0:1, :]

        return cls_output, x, padding_mask

    def forward(
        self,
        hidden_states: dict[str, torch.Tensor],
        attention_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Dict mapping model name to encoder outputs (B, T_k, D_k).
            attention_masks: Dict mapping model name to boolean masks (B, T_k),
                True where valid.

        Returns:
            probs: Predicted probability for each model of shape (B, n_models).
        """
        B = next(iter(hidden_states.values())).size(0)

        cls_outputs = {}
        full_sequences = {}
        full_padding_masks = {}

        for idx, model_name in enumerate(self.model_names):
            cls_out, full_seq, pad_mask = self._run_stage1(
                model_idx=idx,
                model_name=model_name,
                hidden_states=hidden_states[model_name],
                attention_mask=attention_masks[model_name],
            )
            cls_outputs[model_name] = cls_out
            full_sequences[model_name] = full_seq
            full_padding_masks[model_name] = pad_mask

        if self.use_cross_attention_bridge:
            updated_cls = {}
            for model_name in self.model_names:
                other_names = [n for n in self.model_names if n != model_name]
                other_seqs = torch.cat(
                    [full_sequences[n][:, 1:, :] for n in other_names], dim=1
                )
                other_masks = torch.cat(
                    [full_padding_masks[n][:, 1:] for n in other_names], dim=1
                )
                updated_cls[model_name] = self.cross_attention_bridges[model_name](
                    cls_token=cls_outputs[model_name],
                    other_sequences=other_seqs,
                    key_padding_mask=other_masks,
                )
            cls_outputs = updated_cls

        global_cls = self.global_cls.expand(B, -1, -1)
        model_summaries = torch.cat(
            [cls_outputs[name] for name in self.model_names], dim=1
        )

        fusion_input = torch.cat([global_cls, model_summaries], dim=1)
        fusion_output = self.stage2_encoder(fusion_input)

        global_cls_output = fusion_output[:, 0, :]
        logits = self.classifier(global_cls_output)
        probs = F.softmax(logits, dim=-1)

        return probs

    def count_parameters(self) -> dict:
        """Count parameters by component.

        Returns:
            Dict mapping component name to parameter count, including total.
        """
        counts = {}
        counts["projection"] = sum(p.numel() for p in self.projection.parameters())
        counts["pos_encoding"] = 0
        counts["model_embeddings"] = sum(
            p.numel() for p in self.model_embeddings.parameters()
        )
        counts["cls_tokens"] = self.cls_tokens.numel()

        if self.share_stage1_weights:
            counts["stage1_encoder"] = sum(
                p.numel() for p in self.stage1_encoder.parameters()
            )
        else:
            counts["stage1_encoders"] = sum(
                p.numel() for p in self.stage1_encoders.parameters()
            )

        if self.use_cross_attention_bridge:
            counts["cross_attention"] = sum(
                p.numel() for p in self.cross_attention_bridges.parameters()
            )

        counts["stage2_encoder"] = sum(
            p.numel() for p in self.stage2_encoder.parameters()
        )
        counts["classifier"] = sum(
            p.numel() for p in self.classifier.parameters()
        )
        counts["global_cls"] = self.global_cls.numel()
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
