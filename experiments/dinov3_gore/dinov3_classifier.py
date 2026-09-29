import torch
from torch import nn

POOL_HEADS = 6


class AttentionPool(nn.Module):
    """
    CLS-conditioned attention pooling over patch tokens.

    Architecture matches the production training implementation:

        CLS -> query
        patches -> keys / values

    The attention output is a 384-D learned local feature.
    """

    def __init__(self, dim=384, heads=POOL_HEADS, dropout=0.10):
        super().__init__()

        if dim % heads:
            raise ValueError(
                f"dim {dim} is not divisible by heads {heads}"
            )

        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.cls_norm = nn.LayerNorm(dim)
        self.patch_norm = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_token, patch_tokens):
        batch_size, num_tokens, dim = patch_tokens.shape

        cls_normed = self.cls_norm(cls_token)
        patches_normed = self.patch_norm(patch_tokens)

        q = self.q_proj(cls_normed).view(
            batch_size,
            1,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = self.k_proj(patches_normed).view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = self.v_proj(patches_normed).view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.dropout(attention)

        pooled = (
            attention @ v
        ).transpose(1, 2).reshape(batch_size, dim)

        return self.out_proj(pooled)


class DinoV3Classifier(nn.Module):
    """
    Production DINOv3 classifier used by the gore/blood/neutral training run.

    Feature representation:

        raw CLS token       : 384-D
        attention-pooled    : 384-D
        --------------------------------
        concatenated         : 768-D

    Classifier:

        LayerNorm(768)
        Linear(768, 384)
        GELU
        Dropout(0.10)
        Linear(384, 128)
        GELU
        Dropout(0.10)
        Linear(128, num_classes)

    Token layout expected from DINOv3:

        [CLS, register_1, ..., register_N, patch_1, ..., patch_M]
    """

    def __init__(
        self,
        backbone,
        num_classes=3,
        hidden_size=384,
        patch_size=16,
        num_register_tokens=4,
        dropout=0.10,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_registers = num_register_tokens
        self.backbone = backbone

        self.attention_pool = AttentionPool(
            hidden_size,
            heads=POOL_HEADS,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, pixel_values):
        height, width = pixel_values.shape[-2:]

        hidden = self.backbone(
            pixel_values=pixel_values
        ).last_hidden_state

        expected = (
            1
            + self.num_registers
            + (height // self.patch_size)
            * (width // self.patch_size)
        )

        if hidden.shape[1] != expected:
            raise ValueError(
                f"backbone returned {hidden.shape[1]} tokens, "
                f"expected {expected}"
            )

        # DINOv3 token layout:
        #
        # [CLS, registers, patches]
        cls_token = hidden[:, 0, :]

        patch_start = 1 + self.num_registers
        patch_tokens = hidden[:, patch_start:, :]

        if patch_tokens.shape[1] == 0:
            raise ValueError("no patch tokens were found")

        pooled_patches = self.attention_pool(
            cls_token,
            patch_tokens,
        )

        # Production training uses the RAW CLS representation.
        return torch.cat(
            [cls_token, pooled_patches],
            dim=1,
        )

    def forward(self, pixel_values):
        features = self.extract_features(pixel_values)
        return self.classifier(features)


def build_model(
    backbone_path,
    num_classes=3,
    dropout=0.10,
):
    """
    Build the production DINOv3 classifier around a local DINOv3 backbone.

    The backbone is loaded from the supplied local model directory so this
    matches the Kaggle training setup using the mounted DINOv3 checkpoint.
    """

    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        backbone_path,
        local_files_only=True,
    )

    config = backbone.config

    return DinoV3Classifier(
        backbone,
        num_classes=num_classes,
        hidden_size=config.hidden_size,
        patch_size=config.patch_size,
        num_register_tokens=config.num_register_tokens,
        dropout=dropout,
    )
