"""Tes: python -m pytest ml/tests"""
import pytest

torch = pytest.importorskip("torch")

from ml.src.model import ArunDCNN  # noqa: E402


def test_output_shape():
    m = ArunDCNN(num_classes=10)
    assert m(torch.randn(4, 3, 32, 32)).shape == (4, 10)


def test_backward_gives_gradients():
    m = ArunDCNN(num_classes=10)
    loss = torch.nn.functional.cross_entropy(m(torch.randn(8, 3, 32, 32)), torch.randint(0, 10, (8,)))
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_can_overfit_small_batch():
    """Sanity check klasik: model harus mampu menghafal 1 batch kecil."""
    torch.manual_seed(0)
    m = ArunDCNN(num_classes=10, channels=(16, 32), dropout=0.0)
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.2
