import pytest
import torch
import torch.nn as nn

from stable_pretraining.forward import lejepa_forward
from stable_pretraining.losses import LeJEPALoss


@pytest.mark.unit
def test_lejepa_loss_uses_global_views_for_centers():
    # proj shape: [V, N, K] = [3, 2, 2]
    proj = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],  # global view 0
            [[3.0, 0.0], [3.0, 0.0]],  # global view 1
            [[10.0, 0.0], [10.0, 0.0]],  # local view
        ]
    )

    loss_fn = LeJEPALoss(lambda_=0.5)
    _, pred_loss, _ = loss_fn(
        proj, return_components=True, global_view_count=2
    )

    # Centers from global views only: mean of [1, 3] = 2.0
    # Pred loss across all views (mean over V*N*K):
    # total squared error = (1+1+64) * 2 samples = 132
    # mean over 3*2*2 elements = 132 / 12 = 11.0
    assert torch.isclose(pred_loss, torch.tensor(11.0))


@pytest.mark.unit
def test_lejepa_forward_handles_mixed_view_sizes():
    class DummyBackbone(nn.Module):
        def __init__(self, out_dim=4):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Linear(3, out_dim)

        def forward(self, x):
            x = self.pool(x).flatten(1)
            return self.proj(x)

    class DummyModule:
        def __init__(self):
            self.backbone = DummyBackbone(out_dim=4)
            self.projector = nn.Linear(4, 2)
            self.lejepa_loss = LeJEPALoss(lambda_=0.1)
            self.training = True

        def log(self, *args, **kwargs):
            return None

    module = DummyModule()

    views = [
        {"image": torch.randn(2, 3, 224, 224), "label": torch.tensor([0, 1])},
        {"image": torch.randn(2, 3, 98, 98), "label": torch.tensor([0, 1])},
    ]

    out = lejepa_forward(module, views, "train")

    assert out["embedding"].shape == (4, 4)
    assert out["loss"].ndim == 0


@pytest.mark.unit
def test_lejepa_forward_single_view_extracts_embedding():
    class DummyBackbone(nn.Module):
        def forward(self, x):
            # Simulate backbone outputs as tuple with token dimension.
            batch = x.shape[0]
            return (torch.randn(batch, 2, 4),)

    class DummyModule:
        def __init__(self):
            self.backbone = DummyBackbone()
            self.training = False

    module = DummyModule()
    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "label": torch.tensor([0, 1]),
    }

    out = lejepa_forward(module, batch, "validate")

    assert out["embedding"].shape == (2, 4)
    assert torch.equal(out["label"], batch["label"])
