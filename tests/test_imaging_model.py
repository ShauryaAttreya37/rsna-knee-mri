from __future__ import annotations

import pytest

from rsna.imaging_training import build_multiseries_model, configure_backbone
from rsna.submission import TARGET_COLUMNS


def test_target_specific_model_forward_and_backward() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            self.norm = nn.LayerNorm(4)
            self.input_projection = nn.Linear(3, 4)

        def forward(self, images):
            features = self.input_projection(images.mean(dim=(2, 3)))
            for block in self.blocks:
                features = block(features)
            return self.norm(features)

    model = build_multiseries_model(FakeBackbone(), feature_dimension=4)
    summary = configure_backbone(model, unfreeze_last_blocks=1)
    images = torch.rand(5, 3, 8, 8)
    metadata = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0]])

    logits = model(images, (2, 3), metadata, encoder_chunk_size=2)
    logits.sum().backward()

    assert logits.shape == (len(TARGET_COLUMNS),)
    assert torch.isfinite(logits).all()
    assert summary["unfrozen_backbone_blocks"] == 1
    assert all(
        parameter.grad is not None
        for parameter in model.backbone.blocks[-1].parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.backbone.blocks[0].parameters()
    )
