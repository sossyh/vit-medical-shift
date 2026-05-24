import torch
import torch.nn as nn
import timm


def get_model(model_name, num_classes=14, pretrained=True, drop_rate=0.0):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def get_vit(num_classes=14, pretrained=True):
    return get_model("vit_base_patch16_224", num_classes, pretrained, drop_rate=0.1)


def get_resnet(num_classes=14, pretrained=True):
    return get_model("resnet50", num_classes, pretrained)


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model, images, n_passes=20, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    enable_mc_dropout(model)
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(images.to(device))
            preds.append(torch.sigmoid(logits).cpu())
    preds = torch.stack(preds)
    return preds.mean(dim=0), preds.std(dim=0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)