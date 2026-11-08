"""Convert a National checkpoint to Core ML (FP32 + FP16) and write the app label file.

Run on the Mac with the isolated Python 3.11 stack from scripts/replacement/requirements-mac.txt:

    python scripts/national/export.py --checkpoint artifacts/national/runs/effb0/best.pt \
        --out artifacts/national/export-effb0 --deployment iOS18

Then validate per scripts/replacement/apple/README.md before replacing
iDex/Recognition/KantoClassifier.mlpackage and iDex/Recognition/pokemon_labels.json.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from model import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--deployment', choices=['iOS18', 'iOS26', 'iOS27'], default='iOS18')
    a = p.parse_args()
    import coremltools as ct
    if not hasattr(ct.target, a.deployment):
        raise SystemExit(f'{a.deployment} absent in coremltools {ct.__version__}')
    if a.out.exists():
        raise SystemExit('use a new export folder; nothing is overwritten')
    a.out.mkdir(parents=True)
    model, state = load_checkpoint(a.checkpoint)
    model.eval()
    size = state['config']['size']
    torch.manual_seed(20261109)
    example = torch.rand(1, 3, size, size)
    traced = torch.jit.trace(model, example, check_trace=True)
    drift = []
    with torch.inference_mode():
        for x in [example, torch.zeros_like(example), torch.ones_like(example), torch.rand_like(example)]:
            d = float((model(x) - traced(x)).abs().max())
            drift.append(d)
            if d > 1e-4:
                raise ValueError(f'trace drift {d}')
    labels_json = json.dumps(state['labels'], indent=2)
    (a.out / 'pokemon_labels.json').write_text(labels_json)
    results = []
    for name, precision in [('fp32', ct.precision.FLOAT32), ('fp16', ct.precision.FLOAT16)]:
        converted = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=getattr(ct.target, a.deployment),
                               compute_precision=precision,
                               inputs=[ct.ImageType(name='image', shape=example.shape, scale=1 / 255, color_layout=ct.colorlayout.RGB)],
                               outputs=[ct.TensorType(name='logits')])
        converted.short_description = f'iDex National Pokédex classifier ({state["config"]["arch"]}, {len(state["labels"])} classes)'
        converted.author = 'iDex local training'
        converted.license = 'Research candidate: source dataset (CC BY-NC-SA 4.0) and Pokémon rights caveats apply'
        converted.user_defined_metadata['checkpoint_sha256'] = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        converted.user_defined_metadata['manifest_sha256'] = state['manifest_sha256']
        converted.user_defined_metadata['app_labels_sha256'] = hashlib.sha256(labels_json.encode()).hexdigest()
        path = a.out / f'NationalClassifier-{name}.mlpackage'
        converted.save(str(path))
        results.append(dict(precision=name, path=str(path)))
    receipt = dict(platform=platform.platform(), torch=torch.__version__, coremltools=ct.__version__, deployment=a.deployment,
                   size=size, classes=len(state['labels']), trace_max_errors=drift, models=results,
                   next_steps=['run scripts/replacement/apple validation on device',
                               'copy the chosen .mlpackage to iDex/Recognition/KantoClassifier.mlpackage (keep the file name the app loads)',
                               'copy pokemon_labels.json to iDex/Recognition/pokemon_labels.json',
                               'update SelectedModelTests digests and RecognitionTests label count'])
    (a.out / 'export_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
