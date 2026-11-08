"""Fixed-shape classifier for the National Pokédex (1025 classes); normalization embedded.

Mirrors scripts/replacement/model.py so export/validation tooling stays interchangeable,
but takes the class count from the label file instead of a constant.
"""
import json
from pathlib import Path
import torch
from torchvision import models

HERE = Path(__file__).resolve().parent
LABELS = HERE / 'pokemon_labels_national.json'
SOURCE = Path('artifacts/national/sources')

PRETRAINED = {
    'mobilenet_v3_large': ('mobilenet_v3_large-5c1a4163.pth', 'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth'),
    'efficientnet_b0': ('efficientnet_b0_rwightman-7f5810bc.pth', 'https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth'),
    'efficientnet_v2_s': ('efficientnet_v2_s-dd5fe13b.pth', 'https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth'),
}


def load_labels(path=LABELS):
    labels = json.loads(Path(path).read_text())
    ids = [x['id'] for x in labels]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError('labels must be unique and sorted by National Dex id')
    return labels


class Classifier(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def make_model(arch, num_classes, pretrained=False):
    if arch == 'mobilenet_v3_large':
        m = models.mobilenet_v3_large()
    elif arch == 'efficientnet_b0':
        m = models.efficientnet_b0()
    elif arch == 'efficientnet_v2_s':
        m = models.efficientnet_v2_s()
    else:
        raise ValueError(arch)
    if pretrained:
        file = SOURCE / 'torchvision' / PRETRAINED[arch][0]
        state = torch.load(file, map_location='cpu', weights_only=True)
        m.load_state_dict(state, strict=True)
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, num_classes)
    return Classifier(m)


def load_checkpoint(file, labels_path=LABELS):
    checkpoint = torch.load(file, map_location='cpu', weights_only=True)
    labels = load_labels(labels_path)
    if checkpoint['labels'] != labels:
        raise ValueError('checkpoint labels mismatch')
    model = make_model(checkpoint['config']['arch'], len(labels))
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, checkpoint
