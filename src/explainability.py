import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.utils import NIH_LABELS


def get_attention_maps(model, image_tensor, device):
    """
    Extract attention weights from every ViT layer.
    These are the raw numbers the model uses to decide
    which parts of the image to focus on.
    """
    model.eval()
    attention_maps = []
    hooks = []

    def make_hook(i):
        def hook(module, input, output):
            attention_maps.append(output.detach().cpu())
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model(image_tensor.unsqueeze(0).to(device))

    for h in hooks:
        h.remove()

    return attention_maps


def attention_rollout(attention_maps, head_fusion="mean", discard_ratio=0.9):
    """
    Compute attention rollout from raw attention maps.
    Combines all layers into one final heatmap.
    Answers: where is the ViT looking overall?
    """
    result = torch.eye(attention_maps[0].size(-1))

    for attn in attention_maps:
        attn = attn.squeeze(0)

        if head_fusion == "mean":
            attn = attn.mean(0)
        elif head_fusion == "max":
            attn = attn.max(0).values
        elif head_fusion == "min":
            attn = attn.min(0).values

        # discard lowest attention weights
        threshold = attn.view(-1).kthvalue(
            int(discard_ratio * attn.numel())
        ).values
        attn[attn < threshold] = 0

        # add residual connection and normalize
        attn = attn + torch.eye(attn.size(-1))
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn, result)

    # CLS token row, drop CLS position
    mask = result[0, 1:]

    # Dynamic reshape — handles variable token sizes
    n = mask.size(0)
    h = int(n ** 0.5)
    # Find largest h such that h * w = n
    while h >= 1:
        if n % h == 0:
            break
        h -= 1
    w = n // h

    mask = mask.reshape(h, w).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def resize_attention(mask, target_size=224):
    """Resize attention map to match original image size."""
    t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(
        t, size=(target_size, target_size),
        mode="bilinear", align_corners=False
    ).squeeze().numpy()


def get_gradcam(model, image_tensor, target_class, target_layer, device):
    """
    Compute GradCAM heatmap for ResNet (CNN baseline).
    Answers: where is ResNet looking for a specific disease?
    """
    cam_obj = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    cam = cam_obj(
        input_tensor=image_tensor.unsqueeze(0).to(device),
        targets=targets
    )
    return cam[0]


def denormalise(tensor):
    """Convert normalised image tensor back to viewable image."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy()
    return np.clip(std * img + mean, 0, 1)


def overlay_attention(image_np, mask, alpha=0.5, cmap="jet"):
    """Overlay attention heatmap on top of image."""
    heatmap = plt.get_cmap(cmap)(mask)[..., :3]
    return (1 - alpha) * image_np + alpha * heatmap


def plot_attention_comparison(images, masks_id, masks_ood,
                               labels, save_path=None, n=4):
    """
    Side by side comparison of attention maps:
    ID (in-distribution) vs OOD (out-of-distribution).
    This is the key figure for your thesis.
    """
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    for i in range(n):
        img = denormalise(images[i])

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"Image\n{labels[i]}", fontsize=9)

        axes[i, 1].imshow(overlay_attention(img, masks_id[i]))
        axes[i, 1].set_title("Attention (ID)\nIn-distribution", fontsize=9)

        axes[i, 2].imshow(overlay_attention(img, masks_ood[i]))
        axes[i, 2].set_title("Attention (OOD)\nOut-of-distribution", fontsize=9)

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.show()
    return fig


def compute_iou_with_bbox(mask, row, threshold=0.5):
    """
    Compare attention map to ground truth bounding box.
    Answers: is the model looking at the right region?

    Args:
        mask:      (H, W) attention map in [0, 1]
        row:       a row from BBox_List_2017.csv
                   columns: 'Bbox [x', 'y', 'w', 'h]'
        threshold: binarise mask at this value
    Returns:
        iou: float between 0 and 1
    """
    x  = int(row['Bbox [x'])
    y  = int(row['y'])
    w  = int(row['w'])
    h  = int(row['h]'])

    binary = (mask >= threshold).astype(np.uint8)
    gt     = np.zeros_like(binary)
    gt[y:y+h, x:x+w] = 1

    intersection = (binary & gt).sum()
    union        = (binary | gt).sum()
    return float(intersection / union) if union > 0 else 0.0
