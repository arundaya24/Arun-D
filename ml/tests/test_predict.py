"""Tes untuk agregasi prediksi saat inferensi."""
import pytest

torch = pytest.importorskip("torch")

from ml.src.predict import Predictor  # noqa: E402


class FirstPixelClassifier(torch.nn.Module):
    def forward(self, x):
        value = x[:, 0, 0, 0]
        return torch.stack((value, -value), dim=1)


def test_tta_probabilities_are_mirror_invariant():
    model = FirstPixelClassifier()
    image = torch.tensor([[[[2.0, -2.0]], [[0.0, 0.0]], [[0.0, 0.0]]]])

    probabilities = Predictor._tta_probabilities(model, image)
    mirrored_probabilities = Predictor._tta_probabilities(
        model, torch.flip(image, dims=(-1,))
    )

    assert torch.allclose(probabilities, mirrored_probabilities)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
