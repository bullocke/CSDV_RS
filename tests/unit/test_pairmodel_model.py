"""The pair network, its loss and the cell scores."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio  # noqa: F401  (GDAL has to load before torch)

from csdv_core.pairmodel.metrics import (
    AgentCounts,
    CellCounts,
    agent_loss,
    masked_bce_dice,
    to_percent,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from csdv_core.pairmodel.model import SiamesePairNet, adapt_stem  # noqa: E402


@pytest.mark.parametrize("cell_px", [16, 8, 4])
def test_output_is_one_logit_per_cell(cell_px: int) -> None:
    model = SiamesePairNet(encoder="resnet18", cell_px=cell_px).eval()
    a = torch.randn(2, 4, 96, 128)
    b = torch.randn(2, 4, 96, 128)
    with torch.no_grad():
        out = model(a, b)
    assert out.shape == (2, 1, 96 // cell_px, 128 // cell_px)


def test_the_order_of_the_dates_matters() -> None:
    torch.manual_seed(0)
    model = SiamesePairNet(encoder="resnet18").eval()
    a, b = torch.randn(1, 4, 64, 64), torch.randn(1, 4, 64, 64)
    with torch.no_grad():
        assert not torch.allclose(model(a, b), model(b, a))
        joined = model.decode(model.encode(a), model.encode(b))
        assert torch.allclose(joined, model(a, b))


def test_agent_head_gives_one_score_per_agent_and_cell() -> None:
    model = SiamesePairNet(encoder="resnet18", n_agents=6).eval()
    a, b = torch.randn(2, 4, 64, 96), torch.randn(2, 4, 64, 96)
    with torch.no_grad():
        loss, agents = model.forward_agents(a, b)
        assert torch.allclose(loss, model(a, b))
    assert loss.shape == (2, 1, 4, 6)
    assert agents.shape == (2, 6, 4, 6)
    plain = SiamesePairNet(encoder="resnet18").eval()
    with torch.no_grad():
        assert plain.forward_agents(a, b)[1] is None


def test_agent_loss_and_counts() -> None:
    logits = torch.zeros(1, 3, 1, 4)
    logits[0, 0, 0, 0] = 9.0  # says class 0, label 0
    logits[0, 2, 0, 1] = 9.0  # says class 2, label 1
    logits[0, 1, 0, 2] = 9.0  # says class 1, no label
    logits[0, 2, 0, 3] = 9.0  # says class 2, label 2
    agent = torch.tensor([[[0, 1, 255, 2]]], dtype=torch.uint8)
    assert float(agent_loss(logits, agent)) > 1.0
    assert float(agent_loss(logits, torch.full_like(agent, 255))) == 0.0
    counts = AgentCounts(("a", "b", "c"))
    counts.update(logits.argmax(dim=1).numpy(), agent.numpy())
    assert counts.n == 3
    assert counts.accuracy() == pytest.approx(2 / 3)
    assert counts.recall() == {"a": 1.0, "b": 0.0, "c": 1.0}
    assert counts.balanced_accuracy() == pytest.approx(2 / 3)
    assert counts.rows()[1]["as_c"] == 1
    assert (counts + counts).n == 6


def test_stem_takes_four_bands_from_three() -> None:
    weight = torch.randn(64, 3, 7, 7)
    state = {"conv1.weight": weight, "fc.weight": torch.zeros(2, 2)}
    out = adapt_stem(state, 4)
    assert "fc.weight" not in out
    assert out["conv1.weight"].shape == (64, 4, 7, 7)
    assert torch.equal(out["conv1.weight"][:, :3], weight)
    assert torch.allclose(out["conv1.weight"][:, 3], weight.mean(dim=1))


def test_encoder_weights_are_read_from_a_file(tmp_path) -> None:
    import torchvision

    path = tmp_path / "resnet18.pth"
    source = torchvision.models.resnet18(weights=None)
    torch.save(source.state_dict(), path)
    model = SiamesePairNet(encoder="resnet18", weights=path)
    assert torch.equal(model.levels[0][0].conv1.weight, source.layer1[0].conv1.weight)
    assert torch.equal(model.stem[0].weight[:, :3], source.conv1.weight)


def test_loss_leaves_out_ignored_cells() -> None:
    logits = torch.tensor([[[[8.0, -8.0], [8.0, 0.0]]]], requires_grad=True)
    target = torch.tensor([[[1, 0], [255, 255]]], dtype=torch.uint8)
    good = masked_bce_dice(logits, target)
    wrong = masked_bce_dice(logits, torch.tensor([[[0, 1], [255, 255]]]).byte())
    assert float(good) < 0.01 < float(wrong)
    nothing = masked_bce_dice(logits, torch.full((1, 2, 2), 255, dtype=torch.uint8))
    assert float(nothing) == 0.0
    nothing.backward()


def test_cell_counts_against_a_hand_count() -> None:
    counts = CellCounts()
    percent = np.array([90, 80, 60, 40, 30, 10, 255, 70], dtype=np.uint8)
    target = np.array([1, 1, 0, 1, 0, 0, 1, 255], dtype=np.uint8)
    counts.update(percent, target)
    assert (counts.n_pos, counts.n_valid) == (3, 6)
    at = counts.at(0.5)
    assert at["precision"] == pytest.approx(2 / 3)
    assert at["recall"] == pytest.approx(2 / 3)
    assert at["iou"] == pytest.approx(2 / 4)
    # Ranked 90+, 80+, 60-, 40+: precision 1 at recall 1/3 and 2/3, 3/4 at 1.
    assert counts.average_precision() == pytest.approx((1 + 1 + 0.75) / 3)
    threshold, f1 = counts.best_f1()
    assert f1 == pytest.approx(6 / 7)
    assert threshold == pytest.approx(0.31)
    both = counts + counts
    assert both.n_valid == 12
    assert both.average_precision() == pytest.approx(counts.average_precision())
    assert to_percent(np.array([0.0, 0.504, 0.996, 1.0])).tolist() == [0, 50, 100, 100]
