# Model and training-data notices

The packaged 1,025-class EfficientNet-B0 was fine-tuned locally for iDex on the [JJMack/pokemon-classification-gen1-9](https://huggingface.co/datasets/JJMack/pokemon-classification-gen1-9/tree/959ff557b102fc3b4f912d0ca769273be5b9ca7f) dataset (CC BY-NC-SA 4.0, Bulbapedia artwork) via `scripts/national/`; the notices below also cover the earlier 151-class Kanto model. Its architecture and ImageNet initialization use TorchVision; the training images were not all created by iDex contributors. This notice applies to the model distribution and does not relicense the entire application, its fonts, audio, catalog or sprites.

## Model contributions

To the extent the iDex contributors hold rights in their trained-model contributions, those contributions are offered for noncommercial sharing under [CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Preserve attribution to iDex and the sources below, identify modifications, and follow the applicable noncommercial/share-alike terms. This limited offer does not grant rights held by others or declare that every underlying training image was cleared for redistribution. No raw training image is included with the model upload.

## Implementation and initialization

- **TorchVision**, PyTorch contributors, EfficientNet-B0 implementation. The BSD3-Clause notice is retained in [TorchVision-LICENSE.txt](TorchVision-LICENSE.txt).
- Initial weights: `efficientnet_b0_rwightman-7f5810bc.pth`, [official PyTorch download](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth), SHA-256 `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`. The classifier head was changed and the model fine-tuned. TorchVision notes that pretrained weights may carry [training-data-specific terms](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights); the software license alone is not universal image/model clearance.
- EfficientNet architecture: Mingxing Tan and Quoc V. Le, [EfficientNet](https://proceedings.mlr.press/v97/tan19a.html). This citation is lineage, not an endorsement.

## Positive training sources

| Source and frozen revision | Publisher declaration / required caution |
| --- | --- |
| [JJMack/pokemon-classification-gen1-9](https://huggingface.co/datasets/JJMack/pokemon-classification-gen1-9/tree/959ff557b102fc3b4f912d0ca769273be5b9ca7f), revision `959ff557b102fc3b4f912d0ca769273be5b9ca7f` | Card declares CC BY-NC-SA4.0 and attributes images to Bulbapedia. Preserve the noncommercial/share-alike and attribution conditions; underlying character/artwork rights remain separate. |
| [keremberke/pokemon-classification](https://huggingface.co/datasets/keremberke/pokemon-classification/tree/a0b014ffa0bf56b0a490676d298b3d73ca52b8d6), revision `a0b014ffa0bf56b0a490676d298b3d73ca52b8d6` | Card declares Public Domain and cites Robert's Pokédex dataset on Roboflow Universe, exported2022-12-20. This is an uploader declaration, not proof of ownership of each artwork. |
| [sgtsaughter/pokemon-classification-images-151](https://huggingface.co/datasets/sgtsaughter/pokemon-classification-images-151/tree/9d40cca6974fced75a285afbb5e4ad381941485d), revision `9d40cca6974fced75a285afbb5e4ad381941485d` | Card declares Apache2.0; image-level provenance is limited. That declaration does not establish rights to all depicted characters/artwork. |

Inputs were audited/mapped to Kanto species, white-composited, centered/resized, deduplicated/grouped and partitioned before training. These are changes to the source data pipeline. The frozen manifest digest is recorded in the model manifest. Source-card declarations were rechecked at the pinned revisions for this publication. Source data cards, download hashes and raw assets remain local; this repository provides acquisition/reproduction code without republishing the image corpus.

Natural-image diagnostic/corrective experiments also used COCO material. COCO's [terms](https://cocodataset.org/#termsofuse) distinguish annotation/website licensing from photograph copyrights and linked Flickr licenses. The selected checkpoint is the positive-data continuation, not the outlier-exposure branch; diagnostics do not establish a new photography license. Raw COCO images are not uploaded here.

Pokémon names, characters and artwork remain associated with Nintendo, Creatures, GAME FREAK and The Pokémon Company. Dataset uploader labels and this notice do not establish blanket commercial rights. This is an unofficial fan/research project; no affiliation or endorsement is claimed. Existing asset/font/audio notices are retained separately in [catalog attribution](../Catalog/ATTRIBUTION.md).
