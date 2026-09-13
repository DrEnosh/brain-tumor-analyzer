import torch
from torch import nn

from ml.models import Classifier


def gradcam(model: Classifier, x: torch.Tensor, class_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard autograd Grad-CAM on the last feature map. Returns (cam[h, w] >= 0, logits[1, C])."""
    captured = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured["a"] = output

    handle = model.features.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x.detach().clone().requires_grad_(True))
            c = int(logits.argmax(1)) if class_idx is None else class_idx
            logits[0, c].backward()
    finally:
        handle.remove()
    a, g = captured["a"], captured["a"].grad
    cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(dim=1))[0]
    return cam.detach(), logits.detach()


class CamExport(nn.Module):
    """ONNX wrapper: image -> (class probabilities, per-class CAMs, Mahalanobis out-of-distribution score)."""

    def __init__(self, model: Classifier, class_means: torch.Tensor, precision: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("class_means", class_means)
        self.register_buffer("precision", precision)

    def forward(self, image: torch.Tensor):
        a = self.model.features(image)
        pooled = self.model.pool(a).flatten(1)
        logits = self.model.fc(pooled)
        cams = torch.einsum("ck,bkhw->bchw", self.model.fc.weight, a)
        diff = pooled[:, None, :] - self.class_means[None]
        ood_score = ((diff @ self.precision) * diff).sum(dim=2).min(dim=1).values
        return torch.softmax(logits, dim=1), cams, ood_score
