"""Evaluation shared by every model: threshold 0.5, per-image metrics averaged over images.

Pooled (dataset-level) IoU/Dice are stored as secondary metrics only."""
import torch


@torch.no_grad()
def evaluate(model, loader, adapter, device):
    model.eval()
    per = {'iou': [], 'dice': [], 'sen': [], 'spe': []}
    tp_all = fp_all = fn_all = tn_all = 0
    for img, msk in loader:
        img, msk = img.to(device), msk.to(device)
        pred = (torch.sigmoid(adapter.infer(model, img)) > 0.5).float()
        for p, t in zip(pred, msk):
            tp = (p * t).sum().item(); fp = (p * (1 - t)).sum().item()
            fn = ((1 - p) * t).sum().item(); tn = ((1 - p) * (1 - t)).sum().item()
            union = tp + fp + fn
            per['iou'].append(tp / union if union > 0 else 1.0)
            per['dice'].append(2 * tp / (2 * tp + fp + fn) if union > 0 else 1.0)
            per['sen'].append(tp / (tp + fn) if tp + fn > 0 else 1.0)
            per['spe'].append(tn / (tn + fp) if tn + fp > 0 else 1.0)
            tp_all += tp; fp_all += fp; fn_all += fn; tn_all += tn
    res = {k: sum(v) / len(v) for k, v in per.items()}
    res['pooled_iou'] = tp_all / max(tp_all + fp_all + fn_all, 1)
    res['pooled_dice'] = 2 * tp_all / max(2 * tp_all + fp_all + fn_all, 1)
    res['n_images'] = len(per['iou'])
    return res, per['iou']
