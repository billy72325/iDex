# iDex trained image classifier

iDex bundles a locally trained **EfficientNet-V2-S FP32 Core ML model** for all **1,025 National Pokédex species** (Generations I–IX). The complete package is tracked in Git at `KantoClassifier.mlpackage` (the file name is kept so the app's loader is unchanged); a normal clone or GitHub source ZIP includes it. The app performs inference offline and does not download weights or upload photos at runtime.

## Artifact and contract

- Version: `national-v3-efficientnet-v2-s-fp32-6d7e865ab59d`.
- Training checkpoint SHA-256: `6d7e865ab59d9298b15c6e4e29aca811398fd040a68a37b25193f04331681e10` (`scripts/national/train_v2.py` recipe `v2-composite-photo-oe`: v2 trained 40 epochs on RGBA artwork composited onto real backgrounds plus 18,890 web photos and COCO negatives, then v3 warm-started from v2 for 30 epochs with 44,124 model-vetted crawled merchandise photos appended; NVIDIA GB10; validation top-1 0.9759 on 2,779 artwork and 0.9844 on 2,829 photos, negatives max-probability p95 0.115; data manifest SHA-256 `90afe1564b9dd77465d050eba3e5f3f75b4c1ae7aba2dbc3ab1c7966e49dd73f`).
- Packaged weights SHA-256: `4e34318c7149c1d127828e377d358e7b85a8206267f0789b544ee75bb6eae543`.
- Model spec SHA-256: `725fccca45aaa99fb98932a8ca167c10b53dadfa920d464d2d64f292ef07b891`.
- Package: 21,332,540 bytes across three files. Regular Git storage is used, with binary attributes and no LFS filter.
- Implementation/initialization: TorchVision EfficientNet-V2-S, initialized from its ImageNet-1K weights and fine-tuned locally. This is trained for iDex, not a claim that the architecture or every input asset originated here. See [model and dataset notices](MODEL_NOTICE.md).
- Input: `image`, fixed 224×224 RGB. ImageType scales by `1/255`; channelwise ImageNet normalization is embedded exactly once (mean `.485,.456,.406`, std `.229,.224,.225`).
- Output: `logits`, FP32 `[1,1025]` finite scores in exact `pokemon_labels.json` order, already divided by the calibration temperature 1.0551 fitted on the validation split (metadata key `baked_temperature`); no softmax is embedded.
- Label-file SHA-256: `3b794189d9468b353b7508208446d7be428c3841f04d8d47b2a65f898e0d17dc`; IDs 1…1025, National Dex ordering.
- Conversion: coremltools 9.0 from a TorchScript trace (torch 2.9.1), FP32 mlprogram, minimum deployment iOS 18; trace parity on four probe inputs was exact (max abs error 0.0). On-device logit parity against the PyTorch source has not yet been re-run for this model; the previous Kanto model's record is kept below for reference.

The previous 151-class Kanto model (checkpoint `b24cf5301ad9…`, labels `13bd2345fa10…`) remains documented in [model-manifest.json](../../scripts/replacement/model-manifest.json) and in Git history.

Verify the bundled weights from the repository root:

```sh
shasum -a 256 iDex/Recognition/KantoClassifier.mlpackage/Data/com.apple.CoreML/weights/weight.bin
```

## Runtime behavior

Photo import corrects orientation, composites alpha onto white, bounds thumbnail area to approximately4.2 megapixels, and produces a centered square no larger than1200px. The classifier creates a BGRA-backed image feature with RGB model semantics and `scaleFill` at224px. Camera processing rotates the retained frame and takes a centered square. Normalization is not repeated in Swift.

`PokemonClassifier` serializes loading and inference in an actor. Physical devices use `.all`; Simulator uses `.cpuOnly`. The app computes ordinary softmax at temperature1, offers the top three results, and accepts an automatic discovery/narration only when the first score is at least.25 and the first/second raw-logit margin is at least.5. Other results remain best guesses. These heuristics are not a trained unknown detector or calibrated probability. There is no background class; unrelated objects can still be accepted, and Generation VIII–IX species with few training images are the least reliable.

## Validation and limits (151-class Kanto model, for reference)

The FP32 artifact passed171 source-logit fixtures under all four compute-unit policies on both an M1 Max Mac and physical `iPhone19,4` running iOS27.1. Physical checks used unchanged top1 and `abs(error) <= .001 + .0001*abs(reference)`. Apple-prepared RGB bytes matched the canonical inputs. The fixtures are validation artwork/unlabelled natural-image cases, not an independent camera-accuracy test.

The stricter public diagnostic reported96.20% top1, but data exposure/duplicate grouping limits remain and broader public results were approximately tied. The151 labelled numerical fixtures gave146 correct; that small selected set is not a generalization estimate. Real-camera superiority, calibrated unknown rejection, actual operator/backend placement and energy consumption have not been established.

On the physical phone, the selected `.all` route measured warm encoded-bytes-to-result p50/p95 **9.91/11.87ms** and30-second sustained p50/p95 **15.56/17.21ms**. This includes preparation, prediction/logit reading and softmax, and excludes sensor capture, UI, file I/O and loading. The pilot had fixed order and different warm/sustained fixture cycles; it is not a blanket speedup claim. See [performance](../../docs/performance.md).

## Build and reproduce

Xcode27.1 and the existing iDex scheme compile the model automatically. No API key, server, model-hosting service, or separate weight download is required. The model-format minimum iOS18 is compatible with the app's iOS27+ floor; no additional older-phone support is implied.

Training/acquisition, tensor-only checkpoint loading, export and numerical validation tools are in [scripts/replacement](../../scripts/replacement/README.md). Training uses grouped public-data splits, a bounded experiment budget and a fixed151-label head. The trained tensor checkpoint, raw training images, private device reports and local rollback copies are not distributed in this Git commit. Re-exporting requires a matching locally retained or newly trained checkpoint; a clone already contains everything needed to build/run inference.

```sh
python scripts/replacement/export.py --checkpoint /path/to/your/checkpoint.pt \
  --out artifacts/replacement/export --deployment iOS18
xcrun coremlc compile iDex/Recognition/KantoClassifier.mlpackage /tmp/idex-model-compiled
```

Export writes fresh FP32 and FP16 diagnostic packages and never overwrites the app package. Only the verified FP32 package is selected for the app. FP16/mixed diagnostics had policy-dependent numerical failures and are not shipped as alternatives.

Local rollback snapshots and historical audit evidence remain in ignored `artifacts/model-rollback/` and Git history. They are not app resources or uploaded training artifacts. Installing a rebuilt app on a phone is separate from obtaining the source/model; preserve existing app data when deploying.
