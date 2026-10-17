"""Band-limited matching loss on the predictive-residual skips (train-only)."""
import torch
import torch.nn.functional as F


def boundary_band(mask, k=3):
    """Morphological gradient (k x k dilation minus erosion) of the binary mask."""
    m = (mask > 0.5).float()
    dil = F.max_pool2d(m, k, stride=1, padding=k // 2)
    ero = -F.max_pool2d(-m, k, stride=1, padding=k // 2)
    return (dil - ero).clamp(0, 1)


def centered_cosine(p, b, eps=1e-8):
    """1 - normalised cross-covariance between prediction band p and target band b, per sample."""
    pf = p.flatten(1); bf = b.flatten(1)
    pf = pf - pf.mean(1, keepdim=True); bf = bf - bf.mean(1, keepdim=True)
    sim = (pf * bf).sum(1) / (pf.norm(dim=1) * bf.norm(dim=1) + eps)
    return (1.0 - sim).mean()


CRITERIA = {'cos': centered_cosine}


def matching_loss(heads, mask, criterion='cos', k=3):
    """Average over the skip heads of the criterion between sigmoid(head) and the resampled band."""
    band = boundary_band(mask, k)
    fn = CRITERIA[criterion]
    tot = 0.0
    for h in heads:
        b = F.interpolate(band, size=h.shape[-2:], mode='nearest')
        tot = tot + fn(torch.sigmoid(h), b)
    return tot / len(heads)
