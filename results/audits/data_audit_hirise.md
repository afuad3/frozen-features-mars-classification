# HiRISE data audit - Mars orbital image (HiRISE) labeled data set version 3.2

- Zenodo record: 4002935
- Total rows in split file: **64947** (expected released images 64947)
- Original (unaugmented) landmarks: **10815**
- Augmented rows: **54132**
- Classes: **8**
- Unmatched source-id filenames: **0** (examples: [])

## Originals-only landmark split counts
- train: 6997 (expected 6997)
- val:   2025 (expected 2025)
- test:  1793 (expected 1793)
- **Counts match paper/dataset:** YES

## Image properties (sampled)
- Sizes: {(227, 227): 50}
- Modes/channels: {'L': 50}

## Class distribution (originals)
- other: 8802 (81.4%)
- crater: 794 (7.3%)
- swiss cheese: 298 (2.8%)
- slope streak: 267 (2.5%)
- bright dune: 250 (2.3%)
- dark dune: 166 (1.5%)
- spider: 164 (1.5%)
- impact ejecta: 74 (0.7%)

## Source-image count per split
| split   |   n_source_images |
|:--------|------------------:|
| test    |                30 |
| train   |               152 |
| val     |                50 |

## Sample filenames
```
ESP_013049_0950_RED-0067.jpg
ESP_019697_2020_RED-0024.jpg
ESP_015962_1695_RED-0016.jpg
ESP_013049_0950_RED-0118.jpg
ESP_015962_1695_RED-0017.jpg
ESP_017395_1700_RED-0098.jpg
ESP_011348_0950_RED-0005.jpg
ESP_046834_1525_RED-0191.jpg
```